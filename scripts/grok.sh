cd "$(dirname "$0")"/..

python main.py \
--json_path "./data/inputs.json" \
--method "multiagent" \
--backbone "grok"
