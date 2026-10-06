def spawn(fn, args=(), nprocs=None, start_method=None):
    # Single-process mock on CPU
    fn(0, *args)
