import matplotlib.pyplot as plt
import sys

data_file = open("cost_evolution.txt", "r")

xs = []
ys = []
for line in data_file.readlines():
	curr_line_data = []
	for t in line.split():
		try:
			curr_line_data.append(float(t))
		except ValueError:
			try:
				curr_line_data.append(int(t))
			except ValueError:
				pass

	print(curr_line_data)
	if len(curr_line_data) != 2:
		print("Dunno what happened here man...exiting")
		sys.exit(-1)
	xs.append(curr_line_data[0])
	ys.append(curr_line_data[1])

plt.plot(xs, ys)
plt.xlabel("Num iterations")
plt.ylabel("Cost")
plt.title("Cost evolution")
plt.savefig("cost_evolution.png")
