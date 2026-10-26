import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("iris_model.onnx")

print("Входы:")
for inp in sess.get_inputs():
    print(inp.name, inp.shape, inp.type)

print("\nВыходы:")
for out in sess.get_outputs():
    print(out.name, out.shape, out.type)

x = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
res = sess.run(None, {sess.get_inputs()[0].name: x})

print("\nРезультат:")
for i, r in enumerate(res):
    print(i, type(r), getattr(r, "shape", None), r)
