# SAVER JSONL manifest

SAVER uses zero-based, half-open word offsets. Every sample contains all
attached image paths; paths may be absolute or relative to `image_root`.

MRE example:

```json
{"id":"mre-1","tokens":["Ada","works","at","OpenAI"],"images":["post/1-0.jpg","post/1-1.jpg"],"head":[0,1],"tail":[3,4],"relation":"works_for"}
```

MNER example:

```json
{"id":"mner-1","tokens":["New","York","welcomes","Ada"],"images":["post/2-0.jpg","post/2-1.jpg"],"entities":[{"start":0,"end":2,"type":"LOC"},{"start":3,"end":4,"type":"PER"}]}
```

Each dataset directory also needs `labels.json`, mapping labels to contiguous
integer ids. MNER must reserve `NONE` as id `0`.
