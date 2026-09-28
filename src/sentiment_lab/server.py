"""Loopback-only teaching demo with read-only report and inference endpoint."""
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .config import Paths
from .predict import load_selected_model, predict_review


def serve(paths: Paths, port: int = 8765) -> None:
    # 输入项目路径与端口；函数持续阻塞处理请求，直到 Ctrl+C 或服务异常退出。
    # 请求链路：网页 fetch → Handler.do_POST → predict_review → respond JSON。
    # Handler 定义在函数内，可访问当前 name/model/port，使本次服务复用同一模型。
    # 服务启动时加载一次已选 Pipeline，后续请求只做推理，不触发训练或修改模型。
    name, model = load_selected_model(paths)

    class Handler(SimpleHTTPRequestHandler):
        def respond(self, status: int, payload: dict):
            # self 是当前 HTTP 请求的处理器；status 是 200/400/404，payload 为结果或错误字典。
            # Content-Length 按 UTF-8 编码后的字节数计算，不能直接使用含中文字符串的长度。
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            # end_headers 结束 HTTP 头，随后 wfile 写入响应体字节；顺序不能颠倒。
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            # HTTPServer 按请求方法调用 do_POST；其返回值不是响应体，必须显式写响应。
            # 静态 GET 请求由父类处理；POST 仅接受预测接口，不提供写文件或训练入口。
            if self.path != "/api/predict":
                self.respond(404, {"error": "Unknown endpoint"})
                return
            try:
                # 先约束请求体字节数，再读取并解析 JSON；predict_review 另检查文本字符数。
                # 两种限制单位不同，UTF-8 中文等字符可能占用多个字节。
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 50000:
                    raise ValueError("Invalid request size")
                origin = self.headers.get("Origin")
                # 浏览器携带 Origin 时只接受本地页面来源；没有 Origin 的客户端可调用。
                # 此检查约束浏览器跨源请求，并不构成用户认证。
                if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
                    raise ValueError("Only local report requests are allowed")
                payload = json.loads(self.rfile.read(size))
                # rfile 是请求体字节流，read(size) 只读取 Content-Length 声明的字节数。
                # JSON 对象映射为 dict；数组、数字、null 等合法 JSON 仍不符合接口结构。
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                self.respond(200, predict_review(name, model, payload.get("text")))
                # 缺少 text 时 get 返回 None，由 predict_review 的类型检查统一拒绝。
            except (ValueError, UnicodeDecodeError) as error:
                # JSON 格式、请求大小、正文等输入错误作为 400 返回，供前端展示原因。
                self.respond(400, {"error": str(error)})

    # 静态文件根目录限定为 reports，绑定 127.0.0.1 供本机教学演示。
    # ThreadingHTTPServer 可并发处理请求；Ctrl+C 或异常退出后 finally 关闭监听 socket。
    # partial 预先绑定静态目录参数；每个请求仍创建自己的 Handler，模型对象只作只读推理。
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(paths.reports)))
    print(f"Report and prediction demo: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
