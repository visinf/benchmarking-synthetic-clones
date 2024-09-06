   if args.dataset == 'imagenet':
        num_classes = 1000
        data_dir = args.imagenet_root
        n_samples = args.n_samples_imagenet
        resizer = None
    elif args.dataset == 'cifar100':
        num_classes = 100
        data_dir = args.cifar100_root
        n_samples = args.n_samples_cifar
        resizer = Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=False)
    elif args.dataset == 'cifar10':
        num_classes = 10
        data_dir = args.cifar10_root
        n_samples = args.n_samples_cifar
        resizer = Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=False)
    eps = args.eps

    # init wandb
    os.environ['WANDB__SERVICE_WAIT'] = '300'
    wandb_user, wandb_project = None, None
    while True:
        try:
            run_eval = wandb.init(
                project=wandb_project,
                job_type='eval',
                name=f'{"rb" if args.full_benchmark else "aa"}-clip-{args.dataset}-{args.norm}-{eps:.2f}'
                     f'-{args.wandb_id if args.wandb_id is not None else args.pretrained}-{args.blackbox_only}-{args.beta}',
                save_code=True,
                config=vars(args),
                mode='online' if args.wandb else 'disabled'
            )
            break
        except wandb.errors.CommError as e:
            print('wandb connection error', file=sys.stderr)
            print(f'error: {e}', file=sys.stderr)
            time.sleep(1)
            print('retrying..', file=sys.stderr)

    if args.devices != '':
        # set cuda visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    main_device = 0
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Number of GPUs available: {num_gpus}")
    else:
        print("No multiple GPUs available.")

    if not args.blackbox_only:
        attacks_to_run = ['apgd-ce', 'apgd-t']
    else:
        attacks_to_run = ['square']
    print(f'[attacks_to_run] {attacks_to_run}')


    if args.wandb_id not in [None, 'none', 'None']:
        assert args.pretrained in [None, 'none', 'None']
        assert args.clip_model_name in [None, 'none', 'None']
        api = wandb.Api()
        run_train = api.run(f'{wandb_user}/{wandb_project}/{args.wandb_id}')
        clip_model_name = run_train.config['clip_model_name']
        print(f'clip_model_name: {clip_model_name}')
        pretrained = run_train.config["output_dir"]
        if pretrained.endswith('_temp'):
            pretrained = pretrained[:-5]
        pretrained += "/checkpoints/final.pt"
    else:
        clip_model_name = args.clip_model_name
        pretrained = args.pretrained
        run_train = None
    del args.clip_model_name, args.pretrained

    print(f'[loading pretrained clip] {clip_model_name} {pretrained}')

    model, preprocessor_without_normalize, normalize = load_clip_model(clip_model_name, pretrained, args.beta)

    if args.dataset != 'imagenet':
        # make sure we don't resize outside the model as this influences threat model
        preprocessor_without_normalize = transforms.ToTensor()
    print(f'[resizer] {resizer}')
    print(f'[preprocessor] {preprocessor_without_normalize}')

    model.eval()
    model.to(main_device)
    with torch.no_grad():
        # Get text label embeddings of all ImageNet classes
        if not args.template == 'ensemble':
            if args.template == 'std':
                template = 'This is a photo of a {}'
            else:
                raise ValueError(f'Unknown template: {args.template}')
            print(f'template: {template}')
            if args.dataset == 'imagenet':
                texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
            elif args.dataset == 'cifar10':
                texts = [template.format(c) for c in CIFAR10_LABELS]
            text_tokens = open_clip.tokenize(texts)
            embedding_text_labels_norm = []
            text_batches = [text_tokens[:500], text_tokens[500:]] if args.dataset == 'imagenet' else [text_tokens]
            for el in text_batches:
                # we need to split the text tokens into two batches because otherwise we run out of memory
                # note that we are accessing the model directly here, not the CustomModel wrapper
                # thus its always normalizing the text embeddings
                embedding_text_labels_norm.append(
                    model.encode_text(el.to(main_device), normalize=True).detach().cpu()
                )
            model.cpu()
            embedding_text_labels_norm = torch.cat(embedding_text_labels_norm).T.to(main_device)
        else:
            assert args.dataset == 'imagenet', 'ensemble only implemented for imagenet'
            with open('CLIP_eval/zeroshot-templates.json', 'r') as f:
                templates = json.load(f)
            templates = templates['imagenet1k']
            print(f'[templates] {templates}')
            embedding_text_labels_norm = []
            for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values():
                texts = [template.format(c=c) for template in templates]
                text_tokens = open_clip.tokenize(texts).to(main_device)
                class_embeddings = model.encode_text(text_tokens)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                embedding_text_labels_norm.append(class_embedding)
            embedding_text_labels_norm = torch.stack(embedding_text_labels_norm, dim=1).to(main_device)

        assert torch.allclose(
            F.normalize(embedding_text_labels_norm, dim=0),
            embedding_text_labels_norm
        )
        if clip_model_name == 'ViT-B-32':
            assert embedding_text_labels_norm.shape == (512, num_classes), embedding_text_labels_norm.shape
        elif clip_model_name == 'ViT-L-14':
            assert embedding_text_labels_norm.shape == (768, num_classes), embedding_text_labels_norm.shape
        else:
            raise ValueError(f'Unknown model: {clip_model_name}')

    # get model
    model = ClassificationModel(
        model=model,
        text_embedding=embedding_text_labels_norm,
        args=args,
        resizer=resizer,
        input_normalize=normalize,
        logit_scale=args.logit_scale,
    )

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    model_name = None
    # device = [torch.device(el) for el in range(num_gpus)]  # currently only single gpu supported
    device = torch.device(main_device)
    torch.cuda.empty_cache()

    dataset_short = (
        'img' if args.dataset == 'imagenet' else
        'c10' if args.dataset == 'cifar10' else
        'c100' if args.dataset == 'cifar100' else
        'unknown'
    )