import moondream as md
from PIL import Image
import time

#MODEL_PATH = "models/moondream-0_5b-int8.mf.gz"
MODEL_PATH = "models/moondream-2b-int4.mf.gz"
#SITUATION_NAME = "elden_ring_boss_start"
#SITUATION_NAME = "elden_ring_player_death"
#IMAGE_PATH = f"{SITUATION_NAME}_composite.jpg"

data  = [ "elden_ring_boss_start", "elden_ring_player_death" ]


print(f"using Model: {MODEL_PATH}")

start = time.perf_counter()
model = md.vl(model=MODEL_PATH)
end = time.perf_counter()
print(f"Time taken to load the model: {end - start:.4f} seconds")

for i, name in enumerate(data):
	IMAGE_PATH = f"{name}_composite.jpg"	
	print(f"Starting iter: {i} and using image: {IMAGE_PATH}")
	image = Image.open(IMAGE_PATH)
	image = Image.open(IMAGE_PATH).convert("RGB")
	print(f"Image size original {image.size}")
	image = image.resize((100, 100))
	print(f"Image size new {image.size}")

	start = time.perf_counter()
	encoded = model.encode_image(image)
	end = time.perf_counter()
	print(f"Time taken to encode the image: {end - start:.4f} seconds")

	start = time.perf_counter()
	answer = model.query(
    		encoded,
    		"This is a video game screenshot. Classify the scene as one of: PLAYER_DEATH (a death/'YOU DIED' overlay is visible), BOSS_FIGHT (a boss healthbar is visible), or OTHER. Reply with just the label."
	)["answer"]
	print("Answer:", answer)
	end = time.perf_counter()
	print(f"Time taken for iter-{i}: {end - start:.4f} seconds")
