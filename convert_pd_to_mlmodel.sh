python convert_pd_to_mlmodel.py \
  --retrained_graph=graph_files/retrained_graph.pb \
  --strip_retrained_graph=graph_files/stripped_retrained_graph.pb \
  --coreml_model_file=graph_files/retrained_mlmodel.mlmodel
sleep 100