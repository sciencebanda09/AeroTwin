from pipeline import demo_data
from src.training.trainer import train_from_csv
from src.surrogate.model import SurrogateModel


def test_train_and_load(tmp_path) -> None:
    data = tmp_path / "data.csv"
    model = tmp_path / "model.joblib"
    frame = demo_data(5, 12)
    frame.to_csv(data, index=False)
    metrics = train_from_csv(data, model)
    assert model.exists()
    assert metrics["rmse"] >= 0
    loaded = SurrogateModel.load(model)
    assert len(loaded.predict(frame.head())) == 5
    prediction, lower, upper, confidence = loaded.predict_with_uncertainty(frame.head())
    assert confidence == 0.9
    assert ((lower <= prediction) & (prediction <= upper)).all().all()
