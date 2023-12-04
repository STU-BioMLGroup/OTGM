def get_default_config(data_name):
    if data_name in ['HandWritten']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[240, 1024, 1024, 1024, 20],
                arch2=[216, 1024, 1024, 1024, 20],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                num_sample=2000,
                num_classes=10,
                aligned_ratio=0.5,
                seed=0,
                start_dual_prediction=100,
                batch_size=256,
                pre_epoch=500,
                epoch=500,
                update=100,
                lr=1.0e-4,
                alpha=4,
                lambda1=0.01,
                lambda2=0.01,
                pre_name='pre1',
                k=10,
                got=dict(
                    init_epoch=200,
                    update_epoch=150,
                    lr=0.5,
                    alpha=0.1,
                    it=30,
                    tau=2,
                    num_iter=10,
                    seed=10
                )
            ),
            cluster_param=dict(
                dim_subspace=5,
                alpha=0.3,
                ro=1
            )
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                num_sample=4485,
                num_classes=15,
                aligned_ratio=0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,
                pre_epoch=500,
                epoch=500,
                update=100,
                lr=1.0e-4,
                alpha=7,
                lambda1=0.01,
                lambda2=0.1,
                pre_name='pre1',
                k=150,
                got=dict(
                    init_epoch=350,
                    update_epoch=50,
                    lr=0.5,
                    alpha=0.1,
                    it=30,
                    tau=2,
                    num_iter=1,
                    seed=10
                )
            ),
            cluster_param=dict(
                dim_subspace=5,
                alpha=0.3,
                ro=1
            )
        )

    elif data_name in ['BDGP']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1750, 1024, 1024, 1024, 10],
                arch2=[79, 1024, 1024, 1024, 10],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                num_sample=2500,
                num_classes=5,
                aligned_ratio=0.5,
                seed=0,
                start_dual_prediction=100,
                batch_size=256,
                pre_epoch=500,
                epoch=500,
                update=100,
                lr=1.0e-4,
                alpha=30,
                lambda1=0.01,
                lambda2=0.1,
                # pre_name='pre2_2',
                pre_name='pre1',
                k=20,
                got=dict(
                    init_epoch=200,
                    update_epoch=50,
                    lr=0.5,
                    alpha=0.1,
                    it=20,
                    tau=2,
                    num_iter=5,
                    seed=10
                )
            ),
            cluster_param=dict(
                dim_subspace=5,
                alpha=0.3,
                ro=1
            )
        )

    elif data_name in ['Caltech101-7']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],
                arch2=[512, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                num_sample=1474,
                num_classes=7,
                aligned_ratio=0.5,
                seed=0,
                start_dual_prediction=100,
                batch_size=256,
                pre_epoch=500,
                epoch=500,
                update=100,
                lr=1.0e-4,
                alpha=20,
                lambda1=0.1,
                lambda2=0.1,
                pre_name='pre2_1',
                k=20,
                got=dict(
                    init_epoch=150,
                    update_epoch=50,
                    lr=0.5,
                    alpha=0.1,
                    it=20,
                    tau=2,
                    num_iter=5,
                    seed=10
                )
            ),
            cluster_param=dict(
                dim_subspace=5,
                alpha=0.3,
                ro=1
            )
        )

    elif data_name in ['Caltech101-20']:
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                num_sample=2386,
                num_classes=20,
                seed=4,
                aligned_ratio=0.5,
                start_dual_prediction=100,
                feature_learning=200,
                batch_size=256,
                pre_epoch=500,
                epoch=500,
                update=100,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=7,
                lambda1=0.01,
                lambda2=0.01,
                pre_name='pre1',
                k=20,
                got=dict(
                    init_epoch=150,
                    update_epoch=50,
                    lr=0.5,
                    alpha=0.1,
                    it=20,
                    tau=2,
                    num_iter=5,
                    seed=10
                )
            ),
            cluster_param=dict(
                dim_subspace=5,
                alpha=0.3,
                ro=1
            )
        )

    elif data_name in ['Reuters_dim10']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[10, 1024, 1024, 1024, 10],
                arch2=[10, 1024, 1024, 1024, 10],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                num_sample=9379,
                num_classes=6,
                aligned_ratio=0.5,
                seed=0,
                start_dual_prediction=100,
                batch_size=256,
                pre_epoch=500,
                epoch=500,
                update=100,
                lr=1.0e-4,
                alpha=30,
                lambda1=0.1,
                lambda2=1.0,
                pre_name='pre1',
                k=200,
                got=dict(
                    init_epoch=150,
                    update_epoch=50,
                    lr=0.5,
                    alpha=0.1,
                    it=20,
                    tau=2,
                    num_iter=5,
                    seed=10
                )
            ),
        )

    else:
        raise Exception('Undefined data_name')
