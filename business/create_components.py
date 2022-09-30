from loko_extensions.model.components import Arg, Component, save_extensions, Input, Output


model_name = Arg(name="model_name", type="text", label="Model Name", helper="Helper text")
train = Arg(name="train", type="boolean", label="Train Model", description="Helper text")
input = Input(id='input', label='Input', service='collections', to='idoutput')
output = Output(id='idoutput', label='Output')
comp1 = Component(name="My First Component", args=[model_name, train],inputs=[input],outputs=[output])
save_extensions([comp1])
print(model_name)