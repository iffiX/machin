import io
import dill

a=dill.Pickler(io.BytesIO())
a.dump({})