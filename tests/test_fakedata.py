from spudtr.epf import EPOCH_ID, TIME
import spudtr.fake_epochs_data as fake_data


def test__generate():

    epochs_df, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=TIME,
        epoch_id=EPOCH_ID,
    )

    epochs_df, channels = fake_data._generate(
        n_epochs=10,
        n_samples=100,
        n_categories=2,
        n_channels=32,
        time=TIME,
        epoch_id=EPOCH_ID,
        seed=10,
    )

    epochs_df, channels = fake_data._generate(
        n_epochs=10, n_samples=100, n_categories=2, n_channels=32
    )

    epochs_df = fake_data._get_df()
