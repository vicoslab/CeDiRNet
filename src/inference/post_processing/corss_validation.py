class CrossValidationPostProcesser:
    '''
    A wrapper over main inference processing class that enables evaluation in cross-validation mode by sequentally
    processing each fold.

    Class requires one specific main_processer for each dataset_it item since each fold needs to be processed with
    different model. All three objects must be provided as array either in init (main_processer_folds) or in main call
    (dataset_it_folds, centerdir_groundtruth_op_fold).

    '''

    def __init__(self, main_processer_folds):
        self.main_processer_folds = main_processer_folds

        # add this in case any other wrapper needs it
        self.device = main_processer_folds[0].device

    def get_center_model_list(self):
        return self.main_processer_folds[0].get_center_model_list()

    def __call__(self, dataset_it_folds, centerdir_groundtruth_op_fold=None):
        # For cross-validation we require separate model (i.e. processer) for every dataset_it
        assert len(dataset_it_folds) == len(self.main_processer_folds), "Mismatch in the size of main_processer_folds and dataset_it_folds"

        if centerdir_groundtruth_op_fold is None:
            centerdir_groundtruth_op_fold = [None] * len(dataset_it_folds)

        assert len(dataset_it_folds) == len(centerdir_groundtruth_op_fold), "Mismatch in the size of centerdir_groundtruth_op_fold and dataset_it_folds"

        # process for every fold with corresponding main_processer
        fold = 0
        num_folds = len(self.main_processer_folds)
        for main_processer, dataset_it, centerdir_groundtruth_op in zip(self.main_processer_folds, dataset_it_folds, centerdir_groundtruth_op_fold):
            for input, output in main_processer(dataset_it, centerdir_groundtruth_op, tqdm_kwargs=dict(postfix="FOLD=%d/%d" % (fold+1, num_folds) )):
                yield input, output

            # unload model when done
            main_processer.clean_memory()

            fold += 1

    def clean_memory(self):
        for p in self.main_processer_folds:
            p.clean_memory()