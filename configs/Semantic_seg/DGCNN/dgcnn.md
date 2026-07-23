EncoderDecoder3D(
  0.98 M, 100.000% Params, 48.57 GFLOPs, 100.000% FLOPs, 
  (data_preprocessor): Det3DDataPreprocessor(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
  (backbone): DGCNNBackbone(
    0.225 M, 22.914% Params, 23.782 GFLOPs, 48.964% FLOPs, 
    (GF_modules): ModuleList(
      0.026 M, 2.650% Params, 17.239 GFLOPs, 35.492% FLOPs, 
      (0): DGCNNGFModule(
        0.005 M, 0.522% Params, 3.439 GFLOPs, 7.081% FLOPs, 
        (groupers): ModuleList(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (0): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
        )
        (mlps): ModuleList(
          0.005 M, 0.522% Params, 3.439 GFLOPs, 7.081% FLOPs, 
          (0): Sequential(
            0.005 M, 0.522% Params, 3.439 GFLOPs, 7.081% FLOPs, 
            (layer0): ConvModule(
              0.001 M, 0.091% Params, 0.629 GFLOPs, 1.295% FLOPs, 
              (conv): Conv2d(0.001 M, 0.078% Params, 0.503 GFLOPs, 1.036% FLOPs, 12, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.013% Params, 0.084 GFLOPs, 0.173% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(0.0 M, 0.000% Params, 0.042 GFLOPs, 0.086% FLOPs, negative_slope=0.2, inplace=True)
            )
            (layer1): ConvModule(
              0.004 M, 0.431% Params, 2.81 GFLOPs, 5.786% FLOPs, 
              (conv): Conv2d(0.004 M, 0.418% Params, 2.684 GFLOPs, 5.527% FLOPs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.013% Params, 0.084 GFLOPs, 0.173% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(0.0 M, 0.000% Params, 0.042 GFLOPs, 0.086% FLOPs, negative_slope=0.2, inplace=True)
            )
          )
        )
      )
      (1): DGCNNGFModule(
        0.013 M, 1.280% Params, 8.305 GFLOPs, 17.098% FLOPs, 
        (groupers): ModuleList(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (0): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
        )
        (mlps): ModuleList(
          0.013 M, 1.280% Params, 8.305 GFLOPs, 17.098% FLOPs, 
          (0): Sequential(
            0.013 M, 1.280% Params, 8.305 GFLOPs, 17.098% FLOPs, 
            (layer0): ConvModule(
              0.008 M, 0.849% Params, 5.495 GFLOPs, 11.313% FLOPs, 
              (conv): Conv2d(0.008 M, 0.836% Params, 5.369 GFLOPs, 11.054% FLOPs, 128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.013% Params, 0.084 GFLOPs, 0.173% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(0.0 M, 0.000% Params, 0.042 GFLOPs, 0.086% FLOPs, negative_slope=0.2, inplace=True)
            )
            (layer1): ConvModule(
              0.004 M, 0.431% Params, 2.81 GFLOPs, 5.786% FLOPs, 
              (conv): Conv2d(0.004 M, 0.418% Params, 2.684 GFLOPs, 5.527% FLOPs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.013% Params, 0.084 GFLOPs, 0.173% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(0.0 M, 0.000% Params, 0.042 GFLOPs, 0.086% FLOPs, negative_slope=0.2, inplace=True)
            )
          )
        )
      )
      (2): DGCNNGFModule(
        0.008 M, 0.849% Params, 5.495 GFLOPs, 11.313% FLOPs, 
        (groupers): ModuleList(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (0): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
        )
        (mlps): ModuleList(
          0.008 M, 0.849% Params, 5.495 GFLOPs, 11.313% FLOPs, 
          (0): Sequential(
            0.008 M, 0.849% Params, 5.495 GFLOPs, 11.313% FLOPs, 
            (layer0): ConvModule(
              0.008 M, 0.849% Params, 5.495 GFLOPs, 11.313% FLOPs, 
              (conv): Conv2d(0.008 M, 0.836% Params, 5.369 GFLOPs, 11.054% FLOPs, 128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.013% Params, 0.084 GFLOPs, 0.173% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): LeakyReLU(0.0 M, 0.000% Params, 0.042 GFLOPs, 0.086% FLOPs, negative_slope=0.2, inplace=True)
            )
          )
        )
      )
    )
    (FA_module): DGCNNFAModule(
      0.199 M, 20.264% Params, 6.543 GFLOPs, 13.471% FLOPs, 
      (mlps): Sequential(
        0.199 M, 20.264% Params, 6.543 GFLOPs, 13.471% FLOPs, 
        (layer0): ConvModule(
          0.199 M, 20.264% Params, 6.543 GFLOPs, 13.471% FLOPs, 
          (conv): Conv1d(0.197 M, 20.055% Params, 6.442 GFLOPs, 13.264% FLOPs, 192, 1024, kernel_size=(1,), stride=(1,), bias=False)
          (bn): BatchNorm1d(0.002 M, 0.209% Params, 0.067 GFLOPs, 0.138% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(0.0 M, 0.000% Params, 0.034 GFLOPs, 0.069% FLOPs, negative_slope=0.2, inplace=True)
        )
      )
    )
  )
  (decode_head): DGCNNHead(
    0.756 M, 77.086% Params, 24.788 GFLOPs, 51.036% FLOPs, 
    (loss_decode): CrossEntropyLoss(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, avg_non_ignore=False)
    (conv_seg): Conv1d(0.001 M, 0.052% Params, 0.017 GFLOPs, 0.035% FLOPs, 256, 2, kernel_size=(1,), stride=(1,))
    (dropout): Dropout(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, p=0.5, inplace=False)
    (FP_module): DGCNNFPModule(
      0.624 M, 63.611% Params, 20.451 GFLOPs, 42.107% FLOPs, 
      (mlps): Sequential(
        0.624 M, 63.611% Params, 20.451 GFLOPs, 42.107% FLOPs, 
        (layer0): ConvModule(
          0.624 M, 63.611% Params, 20.451 GFLOPs, 42.107% FLOPs, 
          (conv): Conv1d(0.623 M, 63.507% Params, 20.401 GFLOPs, 42.003% FLOPs, 1216, 512, kernel_size=(1,), stride=(1,), bias=False)
          (bn): BatchNorm1d(0.001 M, 0.104% Params, 0.034 GFLOPs, 0.069% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): LeakyReLU(0.0 M, 0.000% Params, 0.017 GFLOPs, 0.035% FLOPs, negative_slope=0.2, inplace=True)
        )
      )
    )
    (pre_seg_conv): ConvModule(
      0.132 M, 13.422% Params, 4.32 GFLOPs, 8.895% FLOPs, 
      (conv): Conv1d(0.131 M, 13.370% Params, 4.295 GFLOPs, 8.843% FLOPs, 512, 256, kernel_size=(1,), stride=(1,), bias=False)
      (bn): BatchNorm1d(0.001 M, 0.052% Params, 0.017 GFLOPs, 0.035% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activate): LeakyReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.017% FLOPs, negative_slope=0.2, inplace=True)
    )
  )
)