def getDataset(args):
    if args.task == 'mri_knee' or args.task == 'mri_brain':
        from datasets.knee_data import kneeData
        data = kneeData(args)
    elif args.task == 'mrf':
        from datasets.mrf import mrfData
        data = mrfData(args)
    elif args.task == 'denoising':
        from datasets.denoising import noisyImages
        data = noisyImages(args)
    elif args.task == 'real_denoising':
        if args.dataset == 'polyu':
           from datasets.denoising import polyU
           data = polyU(args)
        elif args.dataset == 'SIDD':
           from datasets.denoising import SIDD
           data = SIDD(args)
    elif args.task == 'inpainting':
        from datasets.inpainting import inpaintImages
        data = inpaintImages(args)
    elif args.task == 'sr':
        from datasets.sr import blurImages
        data = blurImages(args)
    else:
        raise NotImplementedError("No such dataset.")
    return data
