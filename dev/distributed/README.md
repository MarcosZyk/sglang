# Manual three-node PD launch scripts

These scripts intentionally contain the SGLang arguments directly so they are
easy to edit for experiments.

1. On `10.87.79.111`, run `./launch_server_p.sh`.
2. On both `10.87.79.112` and `10.87.79.113`, run `./launch_server_d.sh`.
3. After all three instances are healthy, run `./launch_lb.sh` on
   `10.87.79.111`.

The load-balanced OpenAI-compatible endpoint is then
`http://10.87.79.111:8000`.

`MC_GID_INDEX=3` selects the routable RoCE v2 GID on these servers.
`--disable-overlap-schedule` avoids the empty timing-queue failure in this
branch's PD overlap path.

