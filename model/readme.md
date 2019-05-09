""mobile_downsample.py""

	base: mobileDownsample: mobilenetV2 with downsample
	main: MGN with linear layer replaced by FC layer
	other: None

"incepmgn"

	base: InceptionMobile: mobilenetV2 combined with Inception
	main: MGN with linear layer replaceed by FC layer
    other: None
	reference: Ning Liu, et al. ADCrowdNet: An Attention-injective Deformable Convolutional Network for Crowd Understanding

"mgnrpp" "pcbrpp"

	base: mobilenetV2
	main: mgn with RPP module
	reference: Yifan Sun, et al. Beyond Part Models: Person Retrieval with Refined Part Pooling(and A Strong Convolutional Baseline)

"mgn"

	base: resnet50
	main: MGN
	reference: Guanshuo Wang, et al. Learning Discriminative Features with Multiple Granularities for Person Re-Identification

"mgn_fuse"

	base: resnet
	reference: None

"mgnmnstn"

	base: mobilenetV2
	main: MGN with spatial transformation network
	reference: Max Jaderberg,et al. Spatial Transformer Networks
