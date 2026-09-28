"""Crawl the publisher's HTML, discover its archive, and stream the reviews.

This is a public dataset acquisition crawler, not a live IMDb review crawler.
The archive is read in place: no untrusted tar paths are extracted to disk.
"""
import hashlib
import logging
import re
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import Paths
from ..io import read_json, write_json

SOURCE = "https://ai.stanford.edu/~amaas/data/sentiment/"
AGENT = "CourseSentimentLab/1.0 (educational public dataset acquisition)"
LOGGER = logging.getLogger(__name__)
# 只匹配有标签的 train/test 评论，四个捕获组依次为划分、情感、评论编号、评分。
# unsup 目录和其他元数据不参与实验；评分仅保留作记录，不作为模型输入。
REVIEW_PATH = re.compile(r"aclImdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt$")


def sha256(path: Path) -> str:
    # 输入本地文件路径，输出 64 位十六进制 SHA-256 字符串；只读文件，不改变缓存。
    # 分块计算整个归档的指纹，避免一次把约 84 MB 文件读入内存。
    # 指纹用于核对本次缓存是否改变，并不代表发布方提供了数字签名。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        # iter(函数, 哨兵) 重复读取，直到读到文件结尾的 b''；lambda 保持每次读取 1 MiB。
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def session() -> requests.Session:
    # 返回配置好的 HTTP 客户端；调用方使用 with session()，结束时关闭连接池。
    # Session 复用连接，并用明确的 User-Agent 标识课程采集程序。
    # 适配器处理请求层重试；下载响应体中途断开的重试另在 download 中处理。
    client = requests.Session()
    client.headers["User-Agent"] = AGENT
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    # 429 为请求过于频繁，5xx 为服务端错误；退避由 urllib3 的 Retry 管理。
    # 此配置安装到 https:// 前缀的适配器，覆盖本项目 Stanford 的 HTTPS 请求。
    client.mount("https://", HTTPAdapter(max_retries=retries))
    return client


def check_robots(client: requests.Session, url: str) -> dict:
    # 输入客户端和目标 URL；允许访问时返回 {'url':robots地址,'status':状态码,'allowed':True}。
    # 禁止访问抛 PermissionError，网络/HTTP 异常继续向上抛，不伪造允许访问记录。
    # robots.txt 位于站点根目录；发布页和数据包 URL 都要分别检查访问规则。
    # timeout 元组依次限制连接等待与读取等待，避免网络异常使流程无限挂起。
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    response = client.get(robots_url, timeout=(15, 60))
    # 本项目将 404 视为未提供 robots 规则；其他 HTTP 错误仍会抛出异常。
    if response.status_code == 404:
        return {"url": robots_url, "status": 404, "allowed": True}
    response.raise_for_status()
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    # parser 依据 User-Agent 和具体目标路径匹配 Allow/Disallow，而非只检查站点域名。
    if not parser.can_fetch("CourseSentimentLab", url):
        raise PermissionError(f"robots.txt disallows collection: {url}")
    return {"url": robots_url, "status": response.status_code, "allowed": True}


def discover_archive(page: str, base_url: str = SOURCE) -> str:
    """Extract the real download link instead of assuming a hardcoded path."""
    # 输入发布页 HTML 字符串，输出目标归档绝对 URL；没有目标链接时抛 ValueError。
    # 例：href='aclImdb_v1.tar.gz' 配合 SOURCE，得到 sentiment 目录下的下载地址。
    # 从真实页面的 a[href] 提取链接，并将相对路径转换为绝对 URL。
    # 只接受发布页同主机的目标归档链接，防止页面中的外站链接被误当成数据源。
    soup = BeautifulSoup(page, "html.parser")
    for link in soup.select("a[href]"):
        # CSS 选择器 a[href] 排除没有 href 的锚点；urljoin 同时支持相对路径和绝对 URL。
        url = urljoin(base_url, link["href"])
        if urlparse(url).path.endswith("/aclImdb_v1.tar.gz"):
            if urlparse(url).hostname != urlparse(base_url).hostname:
                raise ValueError("Unexpected off-site archive URL")
            return url
    raise ValueError("Publisher page has no aclImdb_v1.tar.gz link")


def download(client: requests.Session, url: str, target: Path) -> None:
    """Download bounded HTTP byte ranges; retry failed bodies, not just headers."""
    # 输入客户端、归档 URL 和正式输出路径；成功后文件完整落盘，返回 None。
    # 调用方须先创建 target 的父目录。下载状态主要由以下变量描述：
    # total=HEAD 声明的总字节数，position=已完成字节数，end=当前段最后一个字节索引，
    # expected=本次响应应有字节数，received=实际从响应体写入的字节数。
    # 先写入 .part，所有分段完整后再替换正式文件，避免采集器使用半成品缓存。
    # position 只在当前调用内推进；重新调用本函数会从头下载，不跨进程续传。
    temporary = target.with_suffix(".part")
    metadata = client.head(url, timeout=(15, 60))
    # HEAD 只取元数据，不先下载正文；这里要求服务器提供可转换为整数的 Content-Length。
    metadata.raise_for_status()
    total = int(metadata.headers["Content-Length"])
    chunk_size = 8 * 1024 * 1024
    # 每次请求最多 8 MiB，每次 iter_content 读取最多 1 MiB，分段请求与读取块大小不同。
    position = 0
    etag = metadata.headers.get("ETag")
    try:
        with temporary.open("wb") as stream:
            while position < total:
                # HTTP Range 两端都包含，所以 end 要减 1，长度应为 end-position+1。
                # 例：total=10、分段大小=4 时，范围依次为 0-3、4-7、8-9（字节）。
                end = min(position + chunk_size, total) - 1
                headers = {"Range": f"bytes={position}-{end}"}
                # 若服务器提供 ETag，用 If-Range 约束后续分段来自同一资源版本。
                # 服务端发现资源变化时可能改回 200，下面会拒绝拼接这样的后续响应。
                if etag:
                    headers["If-Range"] = etag
                for attempt in range(3):
                    try:
                        # 上次失败可能已经写入部分响应体；回到本段起点并截断，
                        # 使重试覆盖失败数据，而不是把同一分段重复追加到归档中。
                        stream.seek(position)
                        stream.truncate()
                        # stream=True 让响应体按块读取；with 确保成功/失败都释放响应连接。
                        with client.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
                            response.raise_for_status()
                            if response.status_code == 206:
                                # 206 表示部分内容。范围、总长度必须与 HEAD 和请求一致，
                                # 仅检查状态码不足以确认分段拼接正确。
                                expected_range = f"bytes {position}-{end}/{total}"
                                if response.headers.get("Content-Range") != expected_range:
                                    raise ValueError("Unexpected Content-Range; archive may have changed")
                                expected = end - position + 1
                            elif response.status_code == 200 and position == 0:
                                # 首段允许服务器忽略 Range 并返回完整文件。
                                expected = total  # Server may ignore Range on the first request.
                            else:
                                raise ValueError("Server stopped honoring byte ranges")
                            received = 0
                            for block in response.iter_content(1024 * 1024):
                                # wb 模式写二进制数据，不做文本解码，避免破坏 gzip 归档。
                                stream.write(block)
                                received += len(block)
                            if received != expected:
                                # 即使 HTTP 请求成功，响应体也可能被截断，不能直接发布缓存。
                                raise IOError("Incomplete archive segment")
                        position += received
                        # 仅在范围和长度都通过检查后推进 position，下一段接在已完成部分后面。
                        LOGGER.info("Downloaded %.0f / %.0f MiB", position / 1024**2, total / 1024**2)
                        break
                    except (requests.RequestException, IOError):
                        # 对连接或响应体读取失败最多尝试三次，间隔为 1、2 秒。
                        # 范围不匹配等 ValueError 不在此重试范围内，直接终止下载。
                        if attempt == 2:
                            raise
                        LOGGER.warning("Retrying segment at byte %s", position)
                        time.sleep(2 ** attempt)
                # 成功段之间短暂停顿，避免连续请求给公开数据服务器带来不必要压力。
                time.sleep(0.2)
        # 临时文件位于正式文件同一目录，完整下载后一次替换正式路径。
        temporary.replace(target)
    finally:
        # 成功后临时文件已被移动；失败时删除它，保留正式缓存不受部分数据污染。
        temporary.unlink(missing_ok=True)


def read_reviews(archive: Path) -> pd.DataFrame:
    # 输入 .tar.gz 路径，输出按 id 排序的原始表，列为 id/split/label/rating/text。
    # 例：aclImdb/train/pos/123_9.txt → id='train/pos/123', label=1, rating=9。
    # 标签来自 pos/neg 目录，rating 保存来源评分；后续拟合只接收 text 与 label。
    # 逐个读取 tar 成员的内容，不把归档路径解压到磁盘，避免路径穿越问题。
    rows = []
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            match = REVIEW_PATH.fullmatch(member.name)
            # fullmatch 要求整条成员路径匹配；isfile 排除目录、链接等非普通文件成员。
            if not match or not member.isfile():
                continue
            split, sentiment, review_id, rating = match.groups()
            # 大小上限防止异常成员占用过多内存；严格 UTF-8 解码使编码错误可见。
            if member.size > 2 * 1024 * 1024:
                raise ValueError(f"Review unexpectedly large: {member.name}")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable review: {member.name}")
            rows.append({"id": f"{split}/{sentiment}/{review_id}", "split": split,
                         "label": int(sentiment == "pos"), "rating": int(rating),
                         "text": stream.read().decode("utf-8", errors="strict")})
    frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    # id 同时包含划分和类别，避免不同目录中的相同数字编号被当作相同评论。
    # 排序使读取顺序不依赖 tar 成员排列，有利于后续复现预测顺序和误判示例。
    # 清洗前核对官方有标签数据的结构：两个划分各有正负两类，每组 12,500 条。
    # 若下载到错误版本或归档缺失评论，应在训练前立即失败。
    counts = frame.groupby(["split", "label"]).size().to_dict()
    # counts 的键是 (split,label) 元组，size 数每组记录；同时检查总量、组数、组大小。
    if len(frame) != 50000 or set(counts.values()) != {12500} or len(counts) != 4:
        raise ValueError(f"Unexpected original dataset size/balance: {counts}")
    return frame


def collect(paths: Paths, offline: bool = False) -> pd.DataFrame:
    # 阶段入口：输入目录对象和离线标志，返回 read_reviews 得到的原始评论表。
    # 首次联网运行写归档、source_page.html 和 provenance.json；复用缓存时不更新来源时间。
    # 离线模式仍会计算文件 SHA-256、解析归档并核对数据结构，不等于跳过完整性检查。
    paths.raw.mkdir(parents=True, exist_ok=True)
    archive = paths.raw / "aclImdb_v1.tar.gz"
    provenance_path = paths.raw / "provenance.json"
    # 归档与来源记录同时存在且指纹一致时，联网模式也会直接复用缓存。
    if archive.exists() and provenance_path.exists():
        provenance = read_json(provenance_path)
        if sha256(archive) != provenance["sha256"]:
            # 指纹不符时直接报错，不静默重下载，以免替换了数据却仍沿用旧结果说明。
            raise ValueError("Cached archive checksum mismatch; remove it and reacquire")
        LOGGER.info("Verified cached archive; network access skipped")
    else:
        # 离线模式不会回退到网络；缺少任一缓存文件都要明确提示先完成采集。
        if offline:
            raise FileNotFoundError("Offline mode needs the archive and provenance.json; run collect first")
        with session() as client:
            robots = check_robots(client, SOURCE)
            response = client.get(SOURCE, timeout=(15, 60))
            response.raise_for_status()
            # 保存发布页原文后发现链接，能够复核采集时的页面内容和链接来源。
            (paths.raw / "source_page.html").write_text(response.text, encoding="utf-8")
            url = discover_archive(response.text)
            archive_robots = check_robots(client, url)
            time.sleep(1)  # Be polite: this crawler needs only a handful of requests.
            download(client, url, archive)
        # 保存实际来源、UTC 时间、文件指纹及 robots 检查结果，便于复核数据获取过程。
        provenance = {"source_page": SOURCE, "archive_url": url,
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                      "sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
                      "robots": [robots, archive_robots], "user_agent": AGENT,
                      "acquisition": "publisher HTML discovery + public benchmark archive",
                      "original_labeled_reviews": 50000, "excluded_unlabeled_reviews": 50000}
        # retrieved_at_utc 记录采集时间，archive_bytes 为字节数；记录无标签数据被排除，
        # 避免把归档中额外的 unsup 评论计入实际训练/测试样本量。
        write_json(provenance_path, provenance)
    reviews = read_reviews(archive)
    LOGGER.info("Read %s labeled reviews", len(reviews))
    return reviews
