import matplotlib.pyplot as plt
import numpy as np


x = np.ones(170)
for i in range(170):
    x[i]=i
y1= np.ones(170)
y2= np.ones(170)
y3= np.ones(170)
y4= np.ones(170)
a = np.ones(170)
b = np.ones(170)
wrap = np.ones(170)


# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
# fig, ax = plt.subplots()  # Create a figure and an axes.
# for i in range(170):
#     y1[i] = 100*(.5+.5*np.cos(2*np.pi*(i/170 -1/3)))
#     y2[i] = 100*(.5+.5*np.cos(2*np.pi*(i/170 +0)))
#     y3[i] = 100*(.5+.5*np.cos(2*np.pi*(i/170 +1/3)))
#     a[i] = (y2[i]- y3[i])
#     b[i] = (2*y1[i]-y2[i]-y3[i])
#     wrap[i]= 50*np.arctan2(1.7320508*a[i], b[i])
#     # if wrap[i] < 0:
    #     if a[i] < 0:
    #         wrap[i] += 2*np.pi
    #     else:
    #         wrap[i] += 1 * np.pi

fig, ax = plt.subplots()  # Create a figure and an axes.
for i in range(170):
    y1[i] = 100*(.5+.5*np.cos(np.pi*(i/170 )))
    y2[i] = 100*(.5+.5*np.cos(np.pi*(i/170 -1/2)))
    y3[i] = 100*(.5+.5*np.cos(np.pi*(i/170 -1)))
    y4[i] = 100*(.5+.5*np.cos(np.pi*(i/170 -3/2)))
    a[i] = (y2[i]- y4[i])
    b[i] = (y1[i]-y3[i])
    wrap[i]= 50*np.arctan2(a[i], b[i])
    # if wrap[i] < 0:
    #     if a[i] < 0:
    #         wrap[i] += 2*np.pi
    #     else:
    #         wrap[i] += 1 * np.pi

    
ax.plot(x, y1, label='0')  # Pl ot some data on the axes.
ax.plot(x, y2, label='1')  # Pl ot some data on the axes.
ax.plot(x, y3, label='2')  # Pl ot some data on the axes
ax.plot(x, y4, label='3')  # Pl ot some data on the axes
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.
plt.show()

fig, bx = plt.subplots() 
bx.plot(x, a, label='a')  # Pl ot some data on the axes
bx.plot(x, b, label='b')  # Pl ot some data on the axes
bx.plot(x,wrap, label='wrap')  # Pl ot some data on the axes
bx.set_xlabel('x label')  # Add an x-label to the axes.
bx.set_ylabel('y label')  # Add a y-label to the axes.
bx.set_title("Simple Plot")  # Add a title to the axes.
bx.legend()  # Add a legend.
plt.show()

