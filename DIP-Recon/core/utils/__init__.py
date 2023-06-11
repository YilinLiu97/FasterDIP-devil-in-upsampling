def getForwardm(args):
    if args.task == 'mri_knee' or args.task == 'mri_brain':
        from utils.forwardm import mri_forwardm
        return mri_forwardm
    elif args.task == 'mrf':
        from utils.forwardm import mrf_forwardm
        return mrf_forwardm
    elif args.task == 'denoising' or args.task == 'real_denoising':
        from utils.forwardm import denoising_forwardm
        return denoising_forwardm
    elif args.task == 'inpainting':
        from utils.forwardm import inpainting_forwardm
        return inpainting_forwardm
    elif args.task == 'sr':
        from utils.forwardm import sr_forwardm
        return sr_forwardm
