EncoderDecoder3D(
  1.88 M, 100.000% Params, 58.908 GFLOPs, 100.000% FLOPs, 
  (data_preprocessor): Det3DDataPreprocessor(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
  (backbone): PointNet2SAMSG(
    1.032 M, 54.888% Params, 34.594 GFLOPs, 58.726% FLOPs, 
    (SA_modules): ModuleList(
      1.032 M, 54.888% Params, 34.594 GFLOPs, 58.726% FLOPs, 
      (0): PointSAModuleMSG(
        0.005 M, 0.250% Params, 4.547 GFLOPs, 7.718% FLOPs, 
        (groupers): ModuleList(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (0): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          (1): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
        )
        (mlps): ModuleList(
          0.005 M, 0.250% Params, 4.547 GFLOPs, 7.718% FLOPs, 
          (0): Sequential(
            0.001 M, 0.056% Params, 0.587 GFLOPs, 0.997% FLOPs, 
            (layer0): ConvModule(
              0.0 M, 0.008% Params, 0.084 GFLOPs, 0.142% FLOPs, 
              (conv): Conv2d(0.0 M, 0.006% Params, 0.059 GFLOPs, 0.100% FLOPs, 6, 16, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.002% Params, 0.017 GFLOPs, 0.028% FLOPs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.014% FLOPs, inplace=True)
            )
            (layer1): ConvModule(
              0.0 M, 0.016% Params, 0.168 GFLOPs, 0.285% FLOPs, 
              (conv): Conv2d(0.0 M, 0.014% Params, 0.143 GFLOPs, 0.242% FLOPs, 16, 16, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.002% Params, 0.017 GFLOPs, 0.028% FLOPs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.014% FLOPs, inplace=True)
            )
            (layer2): ConvModule(
              0.001 M, 0.032% Params, 0.336 GFLOPs, 0.570% FLOPs, 
              (conv): Conv2d(0.001 M, 0.029% Params, 0.285 GFLOPs, 0.484% FLOPs, 16, 32, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.003% Params, 0.034 GFLOPs, 0.057% FLOPs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.017 GFLOPs, 0.028% FLOPs, inplace=True)
            )
          )
          (1): Sequential(
            0.004 M, 0.194% Params, 3.959 GFLOPs, 6.721% FLOPs, 
            (layer0): ConvModule(
              0.0 M, 0.015% Params, 0.336 GFLOPs, 0.570% FLOPs, 
              (conv): Conv2d(0.0 M, 0.012% Params, 0.235 GFLOPs, 0.399% FLOPs, 6, 32, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.003% Params, 0.067 GFLOPs, 0.114% FLOPs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.034 GFLOPs, 0.057% FLOPs, inplace=True)
            )
            (layer1): ConvModule(
              0.001 M, 0.060% Params, 1.208 GFLOPs, 2.051% FLOPs, 
              (conv): Conv2d(0.001 M, 0.056% Params, 1.107 GFLOPs, 1.880% FLOPs, 32, 32, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.003% Params, 0.067 GFLOPs, 0.114% FLOPs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.034 GFLOPs, 0.057% FLOPs, inplace=True)
            )
            (layer2): ConvModule(
              0.002 M, 0.119% Params, 2.416 GFLOPs, 4.101% FLOPs, 
              (conv): Conv2d(0.002 M, 0.112% Params, 2.215 GFLOPs, 3.759% FLOPs, 32, 64, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.007% Params, 0.134 GFLOPs, 0.228% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.067 GFLOPs, 0.114% FLOPs, inplace=True)
            )
          )
        )
        (points_sampler): PointsSampler(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (samplers): ModuleList(
            0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
            (0): DFPSSampler(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
      )
      (1): PointSAModuleMSG(
        0.045 M, 2.395% Params, 9.37 GFLOPs, 15.906% FLOPs, 
        (groupers): ModuleList(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (0): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          (1): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
        )
        (mlps): ModuleList(
          0.045 M, 2.395% Params, 9.37 GFLOPs, 15.906% FLOPs, 
          (0): Sequential(
            0.019 M, 1.032% Params, 2.575 GFLOPs, 4.372% FLOPs, 
            (layer0): ConvModule(
              0.007 M, 0.347% Params, 0.864 GFLOPs, 1.467% FLOPs, 
              (conv): Conv2d(0.006 M, 0.340% Params, 0.839 GFLOPs, 1.424% FLOPs, 99, 64, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.007% Params, 0.017 GFLOPs, 0.028% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.014% FLOPs, inplace=True)
            )
            (layer1): ConvModule(
              0.004 M, 0.228% Params, 0.57 GFLOPs, 0.968% FLOPs, 
              (conv): Conv2d(0.004 M, 0.221% Params, 0.545 GFLOPs, 0.926% FLOPs, 64, 64, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.007% Params, 0.017 GFLOPs, 0.028% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.014% FLOPs, inplace=True)
            )
            (layer2): ConvModule(
              0.009 M, 0.456% Params, 1.141 GFLOPs, 1.937% FLOPs, 
              (conv): Conv2d(0.008 M, 0.443% Params, 1.091 GFLOPs, 1.851% FLOPs, 64, 128, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.014% Params, 0.034 GFLOPs, 0.057% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.017 GFLOPs, 0.028% FLOPs, inplace=True)
            )
          )
          (1): Sequential(
            0.026 M, 1.364% Params, 6.795 GFLOPs, 11.535% FLOPs, 
            (layer0): ConvModule(
              0.007 M, 0.347% Params, 1.728 GFLOPs, 2.933% FLOPs, 
              (conv): Conv2d(0.006 M, 0.340% Params, 1.678 GFLOPs, 2.848% FLOPs, 99, 64, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.007% Params, 0.034 GFLOPs, 0.057% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.017 GFLOPs, 0.028% FLOPs, inplace=True)
            )
            (layer1): ConvModule(
              0.006 M, 0.342% Params, 1.711 GFLOPs, 2.905% FLOPs, 
              (conv): Conv2d(0.006 M, 0.332% Params, 1.636 GFLOPs, 2.777% FLOPs, 64, 96, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.010% Params, 0.05 GFLOPs, 0.085% FLOPs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.025 GFLOPs, 0.043% FLOPs, inplace=True)
            )
            (layer2): ConvModule(
              0.013 M, 0.674% Params, 3.355 GFLOPs, 5.696% FLOPs, 
              (conv): Conv2d(0.012 M, 0.661% Params, 3.255 GFLOPs, 5.525% FLOPs, 96, 128, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.014% Params, 0.067 GFLOPs, 0.114% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.034 GFLOPs, 0.057% FLOPs, inplace=True)
            )
          )
        )
        (points_sampler): PointsSampler(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (samplers): ModuleList(
            0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
            (0): DFPSSampler(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
      )
      (2): PointSAModuleMSG(
        0.22 M, 11.720% Params, 10.886 GFLOPs, 18.479% FLOPs, 
        (groupers): ModuleList(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (0): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          (1): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
        )
        (mlps): ModuleList(
          0.22 M, 11.720% Params, 10.886 GFLOPs, 18.479% FLOPs, 
          (0): Sequential(
            0.11 M, 5.860% Params, 3.629 GFLOPs, 6.160% FLOPs, 
            (layer0): ConvModule(
              0.034 M, 1.784% Params, 1.103 GFLOPs, 1.873% FLOPs, 
              (conv): Conv2d(0.033 M, 1.770% Params, 1.091 GFLOPs, 1.851% FLOPs, 259, 128, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.014% Params, 0.008 GFLOPs, 0.014% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.007% FLOPs, inplace=True)
            )
            (layer1): ConvModule(
              0.026 M, 1.366% Params, 0.848 GFLOPs, 1.439% FLOPs, 
              (conv): Conv2d(0.025 M, 1.345% Params, 0.829 GFLOPs, 1.406% FLOPs, 128, 196, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.021% Params, 0.013 GFLOPs, 0.022% FLOPs, 196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.006 GFLOPs, 0.011% FLOPs, inplace=True)
            )
            (layer2): ConvModule(
              0.051 M, 2.710% Params, 1.678 GFLOPs, 2.848% FLOPs, 
              (conv): Conv2d(0.05 M, 2.683% Params, 1.653 GFLOPs, 2.805% FLOPs, 196, 256, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.017 GFLOPs, 0.028% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.014% FLOPs, inplace=True)
            )
          )
          (1): Sequential(
            0.11 M, 5.860% Params, 7.257 GFLOPs, 12.320% FLOPs, 
            (layer0): ConvModule(
              0.034 M, 1.784% Params, 2.206 GFLOPs, 3.745% FLOPs, 
              (conv): Conv2d(0.033 M, 1.770% Params, 2.181 GFLOPs, 3.702% FLOPs, 259, 128, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.014% Params, 0.017 GFLOPs, 0.028% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.014% FLOPs, inplace=True)
            )
            (layer1): ConvModule(
              0.026 M, 1.366% Params, 1.696 GFLOPs, 2.878% FLOPs, 
              (conv): Conv2d(0.025 M, 1.345% Params, 1.657 GFLOPs, 2.813% FLOPs, 128, 196, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.0 M, 0.021% Params, 0.026 GFLOPs, 0.044% FLOPs, 196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.013 GFLOPs, 0.022% FLOPs, inplace=True)
            )
            (layer2): ConvModule(
              0.051 M, 2.710% Params, 3.355 GFLOPs, 5.696% FLOPs, 
              (conv): Conv2d(0.05 M, 2.683% Params, 3.305 GFLOPs, 5.611% FLOPs, 196, 256, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.034 GFLOPs, 0.057% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.017 GFLOPs, 0.028% FLOPs, inplace=True)
            )
          )
        )
        (points_sampler): PointsSampler(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (samplers): ModuleList(
            0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
            (0): DFPSSampler(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
      )
      (3): PointSAModuleMSG(
        0.762 M, 40.522% Params, 9.792 GFLOPs, 16.622% FLOPs, 
        (groupers): ModuleList(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (0): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          (1): QueryAndGroup(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
        )
        (mlps): ModuleList(
          0.762 M, 40.522% Params, 9.792 GFLOPs, 16.622% FLOPs, 
          (0): Sequential(
            0.332 M, 17.636% Params, 2.724 GFLOPs, 4.625% FLOPs, 
            (layer0): ConvModule(
              0.133 M, 7.054% Params, 1.088 GFLOPs, 1.848% FLOPs, 
              (conv): Conv2d(0.132 M, 7.027% Params, 1.082 GFLOPs, 1.837% FLOPs, 515, 256, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.004 GFLOPs, 0.007% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.004% FLOPs, inplace=True)
            )
            (layer1): ConvModule(
              0.066 M, 3.527% Params, 0.545 GFLOPs, 0.926% FLOPs, 
              (conv): Conv2d(0.066 M, 3.500% Params, 0.539 GFLOPs, 0.915% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.004 GFLOPs, 0.007% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.004% FLOPs, inplace=True)
            )
            (layer2): ConvModule(
              0.133 M, 7.054% Params, 1.091 GFLOPs, 1.851% FLOPs, 
              (conv): Conv2d(0.132 M, 7.000% Params, 1.078 GFLOPs, 1.830% FLOPs, 256, 512, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.001 M, 0.054% Params, 0.008 GFLOPs, 0.014% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.007% FLOPs, inplace=True)
            )
          )
          (1): Sequential(
            0.43 M, 22.886% Params, 7.067 GFLOPs, 11.997% FLOPs, 
            (layer0): ConvModule(
              0.133 M, 7.054% Params, 2.177 GFLOPs, 3.695% FLOPs, 
              (conv): Conv2d(0.132 M, 7.027% Params, 2.164 GFLOPs, 3.674% FLOPs, 515, 256, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.008 GFLOPs, 0.014% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.007% FLOPs, inplace=True)
            )
            (layer1): ConvModule(
              0.099 M, 5.291% Params, 1.636 GFLOPs, 2.777% FLOPs, 
              (conv): Conv2d(0.099 M, 5.250% Params, 1.617 GFLOPs, 2.745% FLOPs, 256, 384, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.001 M, 0.041% Params, 0.013 GFLOPs, 0.021% FLOPs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.006 GFLOPs, 0.011% FLOPs, inplace=True)
            )
            (layer2): ConvModule(
              0.198 M, 10.541% Params, 3.255 GFLOPs, 5.525% FLOPs, 
              (conv): Conv2d(0.197 M, 10.486% Params, 3.23 GFLOPs, 5.482% FLOPs, 384, 512, kernel_size=(1, 1), stride=(1, 1))
              (bn): BatchNorm2d(0.001 M, 0.054% Params, 0.017 GFLOPs, 0.028% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (activate): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.014% FLOPs, inplace=True)
            )
          )
        )
        (points_sampler): PointsSampler(
          0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
          (samplers): ModuleList(
            0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
            (0): DFPSSampler(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
      )
    )
    (aggregation_mlps): ModuleList(
      0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
      (0): None
      (1): None
      (2): None
      (3): None
    )
  )
  (decode_head): PointNet2Head(
    0.848 M, 45.112% Params, 24.314 GFLOPs, 41.274% FLOPs, 
    (loss_decode): CrossEntropyLoss(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, avg_non_ignore=False)
    (conv_seg): Conv1d(0.0 M, 0.014% Params, 0.068 GFLOPs, 0.115% FLOPs, 128, 2, kernel_size=(1,), stride=(1,))
    (dropout): Dropout(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, p=0.5, inplace=False)
    (FP_modules): ModuleList(
      0.831 M, 44.206% Params, 19.817 GFLOPs, 33.641% FLOPs, 
      (0): PointFPModule(
        0.46 M, 24.459% Params, 0.943 GFLOPs, 1.600% FLOPs, 
        (mlps): Sequential(
          0.46 M, 24.459% Params, 0.943 GFLOPs, 1.600% FLOPs, 
          (layer0): ConvModule(
            0.394 M, 20.946% Params, 0.807 GFLOPs, 1.370% FLOPs, 
            (conv): Conv2d(0.393 M, 20.918% Params, 0.805 GFLOPs, 1.367% FLOPs, 1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.001 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(0.0 M, 0.000% Params, 0.001 GFLOPs, 0.001% FLOPs, inplace=True)
          )
          (layer1): ConvModule(
            0.066 M, 3.514% Params, 0.136 GFLOPs, 0.231% FLOPs, 
            (conv): Conv2d(0.066 M, 3.486% Params, 0.134 GFLOPs, 0.228% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.001 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(0.0 M, 0.000% Params, 0.001 GFLOPs, 0.001% FLOPs, inplace=True)
          )
        )
      )
      (1): PointFPModule(
        0.198 M, 10.514% Params, 1.623 GFLOPs, 2.755% FLOPs, 
        (mlps): Sequential(
          0.198 M, 10.514% Params, 1.623 GFLOPs, 2.755% FLOPs, 
          (layer0): ConvModule(
            0.132 M, 7.000% Params, 1.08 GFLOPs, 1.833% FLOPs, 
            (conv): Conv2d(0.131 M, 6.973% Params, 1.074 GFLOPs, 1.823% FLOPs, 512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.004 GFLOPs, 0.007% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.004% FLOPs, inplace=True)
          )
          (layer1): ConvModule(
            0.066 M, 3.514% Params, 0.543 GFLOPs, 0.922% FLOPs, 
            (conv): Conv2d(0.066 M, 3.486% Params, 0.537 GFLOPs, 0.911% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.004 GFLOPs, 0.007% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.004% FLOPs, inplace=True)
          )
        )
      )
      (2): PointFPModule(
        0.124 M, 6.578% Params, 4.064 GFLOPs, 6.899% FLOPs, 
        (mlps): Sequential(
          0.124 M, 6.578% Params, 4.064 GFLOPs, 6.899% FLOPs, 
          (layer0): ConvModule(
            0.091 M, 4.821% Params, 2.978 GFLOPs, 5.055% FLOPs, 
            (conv): Conv2d(0.09 M, 4.794% Params, 2.953 GFLOPs, 5.013% FLOPs, 352, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.001 M, 0.027% Params, 0.017 GFLOPs, 0.028% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.014% FLOPs, inplace=True)
          )
          (layer1): ConvModule(
            0.033 M, 1.757% Params, 1.086 GFLOPs, 1.844% FLOPs, 
            (conv): Conv2d(0.033 M, 1.743% Params, 1.074 GFLOPs, 1.823% FLOPs, 256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.0 M, 0.014% Params, 0.008 GFLOPs, 0.014% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.007% FLOPs, inplace=True)
          )
        )
      )
      (3): PointFPModule(
        0.05 M, 2.656% Params, 13.187 GFLOPs, 22.386% FLOPs, 
        (mlps): Sequential(
          0.05 M, 2.656% Params, 13.187 GFLOPs, 22.386% FLOPs, 
          (layer0): ConvModule(
            0.017 M, 0.885% Params, 4.396 GFLOPs, 7.462% FLOPs, 
            (conv): Conv2d(0.016 M, 0.872% Params, 4.295 GFLOPs, 7.291% FLOPs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.0 M, 0.014% Params, 0.067 GFLOPs, 0.114% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(0.0 M, 0.000% Params, 0.034 GFLOPs, 0.057% FLOPs, inplace=True)
          )
          (layer1): ConvModule(
            0.017 M, 0.885% Params, 4.396 GFLOPs, 7.462% FLOPs, 
            (conv): Conv2d(0.016 M, 0.872% Params, 4.295 GFLOPs, 7.291% FLOPs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.0 M, 0.014% Params, 0.067 GFLOPs, 0.114% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(0.0 M, 0.000% Params, 0.034 GFLOPs, 0.057% FLOPs, inplace=True)
          )
          (layer2): ConvModule(
            0.017 M, 0.885% Params, 4.396 GFLOPs, 7.462% FLOPs, 
            (conv): Conv2d(0.016 M, 0.872% Params, 4.295 GFLOPs, 7.291% FLOPs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.0 M, 0.014% Params, 0.067 GFLOPs, 0.114% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activate): ReLU(0.0 M, 0.000% Params, 0.034 GFLOPs, 0.057% FLOPs, inplace=True)
          )
        )
      )
    )
    (pre_seg_conv): ConvModule(
      0.017 M, 0.892% Params, 4.429 GFLOPs, 7.519% FLOPs, 
      (conv): Conv1d(0.017 M, 0.878% Params, 4.329 GFLOPs, 7.348% FLOPs, 128, 128, kernel_size=(1,), stride=(1,))
      (bn): BatchNorm1d(0.0 M, 0.014% Params, 0.067 GFLOPs, 0.114% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activate): ReLU(0.0 M, 0.000% Params, 0.034 GFLOPs, 0.057% FLOPs, inplace=True)
    )
  )
)