import taichi as ti

ti.init(arch=ti.gpu)

n = 1000
pixels = ti.field(dtype=float, shape=(n , n))


@ti.kernel
def paint(pxm_x: int, pxm_y: int):
    for i, j in pixels:  # Parallized over all pixels
        if i > 0 and i < n-1 and j > 0 and j < n-1:
            pixels[i, j] = (pixels[i+1, j] + pixels[i-1, j] + pixels[i, j+1] + pixels[i, j-1])/4
        if pxm_x > 0 and pxm_x < n-1 and pxm_y > 0 and pxm_y < n-1: 
            pixels[pxm_x, pxm_y] = 1


gui = ti.GUI("Heat transfer", res=(n , n))

iters = gui.slider('Iters/Frame', 1, int(n*n/n*2), step=10)

iters = 0
while gui.running:
    iters +=1
    mouse_x, mouse_y = gui.get_cursor_pos()
    pxm_x = int(ti.math.floor(mouse_x*n))
    pxm_y = int(ti.math.floor(mouse_y*n))
    paint(pxm_x, pxm_y)
    
    if iters >= ti.math.floor(iters.value):
        gui.set_image(pixels)
        iters = 0
        gui.show()