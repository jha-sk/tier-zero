# Corpus scan — Server Fault dump (2024-04-07)

```
scanning data/serverfault/Posts.xml (1.21 GB)

rows            844092
  questions     325169
  answers       514457
  other         4466
with accepted   153269  (47.1% of questions)
containing code 266959  (31.6% of rows)
cleaned text    0.72 GB
distinct tags   3920

elapsed         13.1s
throughput      64543 rows/s   92.6 MB/s
peak RSS        6 MB   <- must stay flat as input grows

top tags: linux(38535) windows(17588) apache-2.2(17289) nginx(17207) ubuntu(16565) networking(15657) domain-name-system(12398) centos(10678) active-directory(10215) windows-server-2008(9003) ssh(8965) amazon-web-services(8728)
```

## Memory scaling (the streaming claim)

| rows | peak RSS |
|---|---|
| 200,000 | 5 MB |
| 844,092 (full 1.21 GB) | 6 MB |

4.2x the input for 1.2x the memory. Nothing is buffered.
