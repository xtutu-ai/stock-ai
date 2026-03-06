# stock-ai

## 快速启动（服务 + 刷新今日数据）

```bash
cd tu_share
./start_and_refresh.sh
```

脚本会自动：
- 检查 `service_tu4` 是否已在 `8000` 端口运行；
- 未运行则后台启动 `uvicorn service_tu4:app --host 0.0.0.0 --port 8000`；
- 用上海时区当天日期调用 `/refresh_universe` 更新缓存数据；
- 输出可访问地址。

可选环境变量：
- `TS_PORT`（默认 `8000`）
- `TS_HOST`（默认 `0.0.0.0`）
- `WITH_BASIC`（默认 `1`）

## 前端看板

启动后打开：

- `http://127.0.0.1:8000/`

页面支持：
- 「刷新今日Universe」：触发 `/refresh_today` 更新当天行情缓存；
- 「运行选股脚本」：触发 `/run_selector`，直接执行 `seelect5_enhanced.py` 的选股流程；
- 「读取最近选股结果」：读取 `/selector_result` 展示最近一次选股输出。
