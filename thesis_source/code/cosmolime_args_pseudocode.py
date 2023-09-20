generator_args = dict(
    generator_fun = 'Callable',
    generator_fun_args = 'Optional[dict] = None',
    num_initial_samples = 'int = 5000',
    num_generated_samples_per_iter = 'int = 1000',
    save_root_path = 'Optional[str] = None', 
    save_file_name = 'Optional[str] = None', 
    save_every_iteration = 'bool = False',
    load_root_path = 'Optional[str] = None', 
    load_file_name = 'Optional[str] = None',
    save_to_default_path = 'bool = False'
)

objective_args = dict(
    params = 'dict containing trial parameters',
    conditional_params_updater = 'Optional[Callable[[dict], dict]] = None',
    maximize_score = 'bool = True',
    error_function = 'Optional[Callable] = None',
    score_function = 'Optional[Callable] = None'
)

optimizer_args = dict(
    objective_model = 'Union[BaseObjective, str]',
    objective_args = objective_args,
    val_size = 'Optional[float] = 0.2',
    train_val_split_seed = 'Optional[int] = None',
    create_study_params = 'Optional[dict] = None',
    optimize_study_params = 'Optional[dict] = None'
)

pp_args = dict(
    transformers_list = 'list[Union[list, tuple]]',
    save_root_path = 'Optional[str] = None',
    save_file_name = 'Optional[str] = None',
    save_every_iteration = 'bool = False',
    load_root_path = 'Optional[str] = None',
    load_file_name = 'Optional[str] = None'
)

components_params = dict(
    preprocessing = 'Optional[list[PreprocessingTransformer, pp_args: dict]]',
    optimizer_args = optimizer_args,
    iterations_between_generations = 'int', 
    target = 'float', 
    max_iter = 'int',
    component_name = 'str', 
    save_to_default_path = 'bool = False'
)

component_data_names = 'str | list | tuple'

common_component_params = 'dict'

emulator_args = dict(
    generator_args = generator_args,
    components_params = components_params,
    component_data_names = component_data_names,
    common_component_params = common_component_params
)