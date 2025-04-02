class StudentAdaptor(BasicTrainer):
	def __init__(self,
                 alpha=0.8,
                 recon_lossfunc=nn.MSELoss(),
                 adapting=False,
                 *args, **kwargs):
		super(StudentTrainer, self).__init__(*args, **kwargs)

		self.modality = {'cimg', 'rimg', 'csi', 'center', 'depth', 'pd', 'tag', 'ctr', 'dpt', 'ind'}

        self.alpha = alpha
        self.lambda_ = 1.

        self.recon_lossfunc = recon_lossfunc
        self.sample_mse = nn.MSELoss(reduction='none')
        self.img_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.adv = nn.CrossEntropyLoss()

        self.loss_terms = ('LOSS', 'MU', 'LOGVAR', 'LATENT', 'FEATURE', 'IMG', 'CTR', 'DPT', 'DOM', 'DOM_ACC', 'TG_LOSS', 'TG_FEA', 'TG_LAT', 'TG_CTR', 'TG_DPT')
        self.pred_terms = ('C_GT', 'R_GT',
                           'TR_PRED', 'R_PRED',
                           'TC_PRED', 'SC_PRED',
                           'T_LATENT', 'S_LATENT',
                           'GT_CTR', 'GT_DPT', 
                           'T_CTR', 'T_DPT',
                           'S_CTR', 'S_DPT',
                           'DOM_GT', 'DOM_PRED',
                           'TAG', 'IND')

        # FOR ADAPTING
        self.valid_phases = {
            'source': ValidationPhase(name='source', loader='valid'),
            'target': ValidationPhase(name='target', loader='valid2')
        }
        self.early_stopping_trigger = 'target'

        # self.loss_terms = ('LOSS', 'IMG', 'CTR', 'DPT')

        self.losslog = MyLossCTR(name=self.name,
                              loss_terms=self.loss_terms,
                              pred_terms=self.pred_terms,
                              depth=True)
        
        self.losslog.ctr = ['GT_CTR', 'T_CTR', 'S_CTR']
        self.losslog.dpt = ['GT_DPT', 'T_DPT', 'S_DPT']
        
        self.teacher = Teacher(device=self.device)
        self.student = Student(device=self.device, teacher=self.teacher)
        
        self.models = {
            'imgen' : self.teacher.imgen,
            'cimgde': self.teacher.cimgde,
            'rimgde': self.teacher.rimgde,
            'ctrde': self.teacher.ctrde,
            'csien' : self.student.csien,
            'dmnde' : self.student.dmnde
                }

        self.calculate_losses = {
        'main': self.calculate_loss_main,
        'Feature_extractor': self.calculate_loss_fe,
        'Domain_classifier': self.calculate_loss_da,
        'Target_adaptation': self.calculate_loss_ta,
        }

        self.training_phases = {'Feature_extractor': TrainingPhase(name = 'Feature_extractor',
                                                                   train_module = Feature_extractor_train,
                                                                   eval_module = Feature_extractor_eval,
                                                                   verbose=False,
                                                                   loss_arg={'reverse_feature': True},
                                                                   plot_terms=('LOSS', 'LATENT', 'MU', 'LOGVAR', 'FEATURE', 'IMG', 'CTR', 'DPT')
                                                                   ),
                                'Domain_classifier': TrainingPhase(name = 'Domain_classifier',
                                                                   train_module = Domain_classifier_train,
                                                                   eval_module = Domain_classifier_eval,
                                                                   loss = 'DOM',
                                                                   tolerance=1,
                                                                   conditioned_update=True,
                                                                   verbose=False,
                                                                   loss_arg={'reverse_feature': False},
                                                                   plot_terms=('DOM', 'DOM_ACC')
                                                                   ),
                                'Target_adaptation': TrainingPhase(name = 'Target_adaptation',
                                                                   train_module = Feature_extractor_train,
                                                                   eval_module = Feature_extractor_eval,
                                                                   loss = 'TG_LOSS',
                                                                   tolerance=1,
                                                                   verbose=False,
                                                                   loss_arg={'reverse_feature': False},
                                                                   plot_terms=('TG_LOSS', 'TG_FEA', 'TG_LAT', 'TG_CTR', 'TG_DPT')
                                                                   ),

                                }

        
        self.latent_weight = 0.1
        self.rimg_weight = 1.e-4
        self.center_weight = 40.
        self.depth_weight = 50.
        self.feature_weight = 10

        self.domain_weight = 0.01

        self.target_fea_weight = 10.
        self.target_lat_weight = 20
        self.target_ctr_weight = 40.
        self.target_dpt_weight = 50.