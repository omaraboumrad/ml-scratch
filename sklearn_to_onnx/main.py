import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import onnxruntime as rt

iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

# Convert into ONNX format.

from skl2onnx import to_onnx

model_name = "iris.onnx"
onx = to_onnx(clr, X[:1])
with open(model_name, "wb") as f:
    f.write(onx.SerializeToString())
print(f"generated model '{model_name}'")


# Check the validity of the model

from onnx.checker import check_model

check_model(model_name, full_check=True, check_custom_domain=True)


# Compute the prediction with onnxruntime.
print(f"using model '{model_name}'")
sess = rt.InferenceSession(model_name, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
print("input_name:", input_name)
label_name = sess.get_outputs()[0].name
print("label_name:", label_name)
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
print("prediction:", pred_onx)

