
def create_model(opt):
    model = None
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'recycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet')
        from .recycle_gan_model import RecycleGANModel
        model = RecycleGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    return model
