#debug
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=AnySat_JEPA GPU_TYPE=H100 GPU=1 NAME=Geom-ASt-dev DEV=True
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=CAPI model.loss.num_classes=512 GPU_TYPE=H100 GPU=1 NAME=GEOm-ASt-CAPI-sinkhorn DEV=True

#Run all losses SA decoder
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=AnySat_JEPA GPU_TYPE=H100 GPU=4 NAME=GEOm-ASt-Original
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=AnySat_JEPA_no_con GPU_TYPE=H100 GPU=4 NAME=GEOm-ASt-NoContra
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=LatentMIM GPU_TYPE=H100 GPU=4 NAME=GEOm-ASt-LatentMiM_t0.2
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=CAPI model.loss.num_classes=512,1024,4096 GPU_TYPE=H100 GPU=4 NAME=GEOm-ASt-CAPI-sinkhorn
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=CAPI model.loss.num_classes=512,1024,4096 model.network.target_head.target=softmax GPU_TYPE=H100 GPU=4 NAME=GEOm-ASt-CAPI_softmax

#Run all losses with CA decoder
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi model.network.instance.predictor._target_=models.networks.encoder.Transformer.CrossAttentionPredictorMulti trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=AnySat_JEPA GPU_TYPE=H100 GPU=4 NAME=GEOm-ASt-Original-CADecoder
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi model.network.instance.predictor._target_=models.networks.encoder.Transformer.CrossAttentionPredictorMulti trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=AnySat_JEPA_no_con GPU_TYPE=H100 GPU=4 NAME=GEOm-ASt-NoContra-CADecoder
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi model.network.instance.predictor._target_=models.networks.encoder.Transformer.CrossAttentionPredictorMulti trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=LatentMIM GPU_TYPE=H100 GPU=4 NAME=GEOm-ASt-LatentMiM-CADecoder
JZAnyrun src/train.py exp=GeoPlex_AnySat model/network/encoder=Any_Tiny_multi model.network.instance.predictor._target_=models.networks.encoder.Transformer.CrossAttentionPredictorMulti trainer.trainer.strategy='ddp_find_unused_parameters_true' model/loss=CAPI GPU_TYPE=H100 GPU=4 NAME=GEOm-ASt-CAPI-CADecoder
