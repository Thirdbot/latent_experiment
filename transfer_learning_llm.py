# transfer learning from geoscience model to student ,so the architecture doesn't need to change or use geoGpt 72B decoder only
# transfer learning from seisBert to qwen vision model

#steps: 1.train_with_label(transfer vision) 2.transfer_learning(text) 3.train_with_no_label(vision+text alignment) 