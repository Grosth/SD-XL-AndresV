import torch  # Ejecutar operaciones matemáticas del modelo
from diffusers import AutoPipelineForText2Image

# Diffusers es una librería especializada en modelos generativos

print("Cargando el modelo...")

# Se recomienda float16 para acelerar la generación y reducir consumo de VRAM en CUDA
modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,  # float16 o float32 (float16 consume menos VRAM)
    variant="fp16",  # Descarga directamente los pesos optimizados en fp16
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear: ")

print("Creando imagen...")

imagen = modelo(
    prompt=prompt,  # Instrucciones o detalle para crear la imagen
    num_inference_steps=25,  # Pasos de reducción de ruido para refinar la imagen
    guidance_scale=7.0,  # Qué tan fiel debe ser al prompt
    height=1024,  # Se agregó la coma faltante arriba
    width=1024,
).images[0]

imagen.save("imagen.png")

print("Imagen guardada.")
