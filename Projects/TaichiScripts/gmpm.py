import taichi as ti

ti.init(arch=ti.cuda)
# sim setup
dt = 1e-4
dim = 3
domain_size = 256
dx, inv_dx = 1 / domain_size, float(domain_size)

block_size = 4
block_num = domain_size / block_size

particle_num = 775196
max_particle_size = 64
# material
p_vol, p_rho = (dx * 0.5)**3, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / \
    ((1+nu) * (1 - 2 * nu))  # Lame parameters

# grid layout
grid_m = ti.var(dt=ti.f32)
grid_v = ti.Vector(3, dt=ti.f32)
# particle layout
par_pos = ti.Vector(3, dt=ti.f32)
par_m = ti.var(ti.f32)
par_v = ti.Vector(3, dt=ti.f32)
par_F = ti.Matrix(3, 3, dt=ti.f32)
# layout define
@ti.layout
def buffers():
    ti.root.dense(ti.ijk, (block_num, block_num, block_num)).dense(ti.ijk,
                                                                   (block_size, block_size, block_size)).place(grid_m)
    for d in ti.static(range(3)):
        ti.root.dense(ti.ijk, (block_num, block_num, block_num)).dense(ti.ijk,
                                                                       (block_size, block_size, block_size)).place(grid_v(d))
        ti.root.dynamic(ti.i).place(par_pos(d))
    ti.root.dense(ti.ijk, (block_num, block_num, block_num)).dense(ti.ijk,
                                                                   (block_size, block_size, block_size)).dynamic(ti.i,
                                                                                                                 max_particle_size).place(par_m)
    for d in ti.static(range(3)):
        ti.root.dense(ti.ijk, (block_num, block_num, block_num)).dense(ti.ijk,
                                                                       (block_size, block_size, block_size)).dynamic(ti.i,
                                                                                                                     max_particle_size).place(par_v(d))
    for i, j in ti.static(ndrange(3, 3)):
        ti.root.dense(ti.ijk, (block_num, block_num, block_num)).dense(ti.ijk,
                                                                       (block_size, block_size, block_size)).dynamic(ti.i,
                                                                                                                     max_particle_size).place(par_F(i, j))


@ti.kernel
def initialize():
    for i in range(particle_num):
        par_pos[i] = ti.Vector([0, 0, 0])

# with open("twodragons.bin", 'rb') as f:
