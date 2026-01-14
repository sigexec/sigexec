```mermaid
flowchart TD
    start([Start])
    node1[Generate]
    start --> node1
    node2[Filter]
    node1 --> node2
    node3[Fft]
    node2 --> node3
    node4[Power]
    node3 --> node4
    node5([End])
    node4 --> node5
```
