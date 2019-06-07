box_np_ops: 封装了numpy和c的一些算法实现，主要是为了把复杂的变换放在一起，这样子其他类就不必去看那些变换，而注重类之间的关系和接口的实现

1. anchor_generator: 空间固定anchors
2. box_coders: anchors -> random boxes
3. region_similarity: 判断box和ground truth的相似度，从而可以计算offset。具体实现参见box_np_ops
4. 
