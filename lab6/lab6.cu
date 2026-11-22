#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

struct Vec3 { float x, y, z; };
struct Body { float x, y, z, q; };
struct Params {
    float a = 15.0f, eps = 0.001f, w = 0.99f, K = 50.0f, dt = 0.01f;
    float g = 20.0f, radius = 0.35f, shiftz = 0.75f;
};
struct Bullet {
    Vec3 position = {0.0f, 0.0f, 0.0f}, velocity = {0.0f, 0.0f, 0.0f};
    float charge = 50.0f;
    bool active = false;
};
struct Pixel { unsigned char r, g, b, a; };

__host__ __device__ Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ Vec3 mul(Vec3 v, float k) { return {v.x * k, v.y * k, v.z * k}; }
float length(Vec3 v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
__device__ Vec3 position(Body p) { return {p.x, p.y, p.z}; }
Vec3 direction(float yaw, float pitch) {
    return {cosf(yaw) * cosf(pitch), sinf(yaw) * cosf(pitch), sinf(pitch)};
}
Bullet shoot(Vec3 camera, float yaw, float pitch) {
    Bullet bullet;
    bullet.velocity = mul(direction(yaw, pitch), 10.0f);
    bullet.position = add(camera, bullet.velocity);
    bullet.active = true;
    return bullet;
}
void advance(Bullet& bullet, float dt) {
    if (bullet.active) bullet.position = add(bullet.position, mul(bullet.velocity, dt));
}

__device__ Vec3 charge_force(Vec3 point, Vec3 source, float charge, float eps) {
    Vec3 delta = sub(point, source);
    float r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    return mul(delta, charge / (r2 * sqrtf(r2) + eps));
}
__device__ float wall_force(float delta, float charge, float eps) {
    float d = fabsf(delta);
    return charge * delta / (d * d * d + eps);
}
__device__ Vec3 external_acceleration(Body p, Vec3 camera, float cameraCharge,
                              Bullet bullet, Params params) {
    Vec3 force = {
        wall_force(p.x - params.a, p.q, params.eps) + wall_force(p.x + params.a, p.q, params.eps),
        wall_force(p.y - params.a, p.q, params.eps) + wall_force(p.y + params.a, p.q, params.eps),
        wall_force(p.z - 2.0f * params.a, p.q, params.eps) + wall_force(p.z, p.q, params.eps)
    };
    force = add(force, charge_force(position(p), camera, cameraCharge, params.eps));
    if (bullet.active)
        force = add(force, charge_force(position(p), bullet.position, bullet.charge, params.eps));
    Vec3 result = mul(force, params.K * p.q);
    result.z -= params.g;
    return result;
}
__device__ Vec3 acceleration(Body p, const Body* all, int n, Vec3 camera,
                     float cameraCharge, Bullet bullet, Params params) {
    Vec3 pairForce = {0.0f, 0.0f, 0.0f};
    for (int j = 0; j < n; ++j)
        pairForce = add(pairForce, charge_force(position(p), position(all[j]), all[j].q, params.eps));
    return add(mul(pairForce, params.K * p.q),
               external_acceleration(p, camera, cameraCharge, bullet, params));
}

__device__ void contain(float& coordinate, float& velocity, float low, float high) {
    if (coordinate < low) {
        coordinate = low;
        if (velocity < 0.0f) velocity *= -0.5f;
    } else if (coordinate > high) {
        coordinate = high;
        if (velocity > 0.0f) velocity *= -0.5f;
    }
}
__device__ void integrate(Body p, Vec3 velocity, Vec3 accel, Params params,
                  Body& next, Vec3& nextVelocity) {
    nextVelocity = add(mul(velocity, params.w), mul(accel, params.dt));
    Vec3 point = add(position(p), mul(nextVelocity, params.dt));
    contain(point.x, nextVelocity.x, -params.a + params.radius, params.a - params.radius);
    contain(point.y, nextVelocity.y, -params.a + params.radius, params.a - params.radius);
    contain(point.z, nextVelocity.z, params.radius, 2.0f * params.a - params.radius);
    next = {point.x, point.y, point.z, p.q};
}

__device__ float field_coordinate(int index, int size, float a) {
    return (2.0f * index / (size - 1) - 1.0f) * a;
}
__device__ float field_contribution(float x, float y, Vec3 source, float charge, Params params) {
    float dx = source.x - x, dy = source.y - y, dz = source.z - params.shiftz;
    return charge / (dx * dx + dy * dy + dz * dz + params.eps);
}
__device__ float field_at(float x, float y, const Body* all, int n, Bullet bullet, Params params) {
    float field = 0.0f;
    for (int j = 0; j < n; ++j)
        field += field_contribution(x, y, position(all[j]), all[j].q, params);
    if (bullet.active)
        field += field_contribution(x, y, bullet.position, bullet.charge, params);
    return params.K * field;
}
std::vector<Body> initial_positions(int n, unsigned seed, Params params) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> horizontal(-params.a + params.radius, params.a - params.radius);
    std::uniform_real_distribution<float> vertical(params.radius, 2.0f * params.a - params.radius);
    std::vector<Body> bodies(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        bodies[i] = {horizontal(generator), horizontal(generator), vertical(generator), 1.0f};
    return bodies;
}


struct Vertex { float u, v; Vec3 normal, point; };
const int rings = 8, slices = 12;
Vertex sphere_vertex(int row, int column) {
    float theta = 3.14159265f * row / rings;
    float phi = 2.f * 3.14159265f * column / slices;
    Vec3 normal = {sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta)};
    return {float(column) / slices, float(row) / rings, normal, normal};
}
std::vector<Vertex> sphere_mesh() {
    std::vector<Vertex> mesh;
    for (int row = 0; row < rings; ++row)
        for (int column = 0; column < slices; ++column) {
            Vertex a = sphere_vertex(row, column), b = sphere_vertex(row + 1, column);
            Vertex c = sphere_vertex(row, column + 1), d = sphere_vertex(row + 1, column + 1);
            mesh.insert(mesh.end(), {a, b, c, c, b, d});
        }
    return mesh;
}

void fail(const char* message) {
    fprintf(stderr, "ERROR: %s\n", message);
    std::exit(1);
}
void check(cudaError_t error) {
    if (error != cudaSuccess) fail(cudaGetErrorString(error));
}

__global__ void step_kernel(const Body* input, Body* output, Vec3* velocities, int n,
                           Vec3 camera, float cameraCharge, Bullet bullet, Params params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec3 force = acceleration(input[i], input, n, camera, cameraCharge, bullet, params);
    integrate(input[i], velocities[i], force, params, output[i], velocities[i]);
}

__global__ void field_kernel(uchar4* output, int width, int height, const Body* bodies,
                            int n, Bullet bullet, Params params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float field = field_at(field_coordinate(x, width, params.a),
                           field_coordinate(y, height, params.a), bodies, n, bullet, params);
    unsigned char blue = static_cast<unsigned char>(fminf(field, 255.0f));
    output[y * width + x] = make_uchar4(0, 0, blue, 255);
}

__global__ void vertices_kernel(Vertex* output, const Vertex* sphere, int meshSize,
                               const Body* bodies, int n, Bullet bullet, Params params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= meshSize * (n + (bullet.active ? 1 : 0))) return;
    int particle = i / meshSize;
    Vec3 center = particle < n ? position(bodies[particle]) : bullet.position;
    float radius = params.radius * (particle < n ? 1.f : 1.5f);
    Vertex v = sphere[i % meshSize];
    v.point = add(center, mul(v.point, radius));
    output[i] = v;
}
const int N = 150, textureSize = 256;
Params params;
Body *bodies = nullptr, *nextBodies = nullptr;
Vec3* velocities = nullptr;
Vertex* sphere = nullptr;
int sphereSize = 0;
GLuint vertexBuffer = 0, pixelBuffer = 0, floorTexture = 0, ballTexture = 0;
cudaGraphicsResource* resources[2] = {};
Vec3 camera = {38.f, -46.f, 34.f};
float yaw = 2.2616f, pitch = -.38f;
Bullet bullet;
bool keys[256] = {}, paused = false, looking = false;
int width = 1100, height = 800, mouseX = 0, mouseY = 0;

void reset() {
    std::vector<Body> initial = initial_positions(N, 6, params);
    check(cudaMemcpy(bodies, initial.data(), N * sizeof(Body), cudaMemcpyHostToDevice));
    check(cudaMemset(velocities, 0, N * sizeof(Vec3)));
    bullet.active = false;
}

GLuint make_texture(int w, int h, const Pixel* pixels) {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return texture;
}

void initialize() {
    unsigned count = 0;
    int device = 0;
    check(cudaGLGetDevices(&count, &device, 1, cudaGLDeviceListAll));
    if (!count) fail("OpenGL must run on an NVIDIA CUDA device");
    check(cudaSetDevice(device));
    fprintf(stderr, "OpenGL: %s\n", glGetString(GL_RENDERER));
    check(cudaMalloc(&bodies, N * sizeof(Body)));
    check(cudaMalloc(&nextBodies, N * sizeof(Body)));
    check(cudaMalloc(&velocities, N * sizeof(Vec3)));
    reset();

    std::vector<Vertex> mesh = sphere_mesh();
    sphereSize = static_cast<int>(mesh.size());
    check(cudaMalloc(&sphere, sphereSize * sizeof(Vertex)));
    check(cudaMemcpy(sphere, mesh.data(), sphereSize * sizeof(Vertex), cudaMemcpyHostToDevice));

    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, (N + 1) * sphereSize * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);
    glGenBuffers(1, &pixelBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, textureSize * textureSize * sizeof(Pixel), nullptr, GL_DYNAMIC_DRAW);
    check(cudaGraphicsGLRegisterBuffer(&resources[0], vertexBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
    check(cudaGraphicsGLRegisterBuffer(&resources[1], pixelBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    floorTexture = make_texture(textureSize, textureSize, nullptr);
    Pixel checker[32 * 32];
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x) {
            unsigned char c = ((x / 4 + y / 4) % 2) ? 240 : 150;
            checker[y * 32 + x] = {c, c, c, 255};
        }
    ballTexture = make_texture(32, 32, checker);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glClearColor(.025f, .035f, .055f, 1.f);
}

void cleanup() {
    for (auto resource : resources)
        if (resource) cudaGraphicsUnregisterResource(resource);
    cudaFree(bodies);
    cudaFree(nextBodies);
    cudaFree(velocities);
    cudaFree(sphere);
    glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &pixelBuffer);
    glDeleteTextures(1, &floorTexture);
    glDeleteTextures(1, &ballTexture);
}

void update_buffers() {
    check(cudaGraphicsMapResources(2, resources));
    Vertex* vertices;
    uchar4* pixels;
    size_t bytes;
    check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vertices), &bytes, resources[0]));
    check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pixels), &bytes, resources[1]));
    int vertexCount = (N + (bullet.active ? 1 : 0)) * sphereSize;
    vertices_kernel<<<(vertexCount + 255) / 256, 256>>>(vertices, sphere, sphereSize, bodies, N, bullet, params);
    check(cudaGetLastError());
    dim3 block(16, 16), grid((textureSize + 15) / 16, (textureSize + 15) / 16);
    field_kernel<<<grid, block>>>(pixels, textureSize, textureSize, bodies, N, bullet, params);
    check(cudaGetLastError());
    check(cudaGraphicsUnmapResources(2, resources));
    glBindTexture(GL_TEXTURE_2D, floorTexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureSize, textureSize, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void display() {
    update_buffers();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60., double(width) / height, .05, 1000.);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    Vec3 forward = direction(yaw, pitch);
    gluLookAt(camera.x, camera.y, camera.z, camera.x + forward.x,
              camera.y + forward.y, camera.z + forward.z, 0, 0, 1);

    float a = params.a;
    glEnable(GL_TEXTURE_2D);
    glColor3f(1, 1, 1);
    glBindTexture(GL_TEXTURE_2D, floorTexture);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex3f(-a, -a, 0);
    glTexCoord2f(1, 0); glVertex3f(a, -a, 0);
    glTexCoord2f(1, 1); glVertex3f(a, a, 0);
    glTexCoord2f(0, 1); glVertex3f(-a, a, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glColor3f(.55f, .65f, .75f);
    glPushMatrix();
    glTranslatef(0, 0, a);
    glutWireCube(2 * a);
    glPopMatrix();

    GLfloat light[] = {-0.4f, -0.6f, 1.f, 0.f};
    glLightfv(GL_LIGHT0, GL_POSITION, light);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, ballTexture);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glInterleavedArrays(GL_T2F_N3F_V3F, sizeof(Vertex), nullptr);
    glColor3f(.4f, 1.f, .55f);
    glDrawArrays(GL_TRIANGLES, 0, N * sphereSize);
    if (bullet.active) {
        glColor3f(1.f, .35f, .12f);
        glDrawArrays(GL_TRIANGLES, N * sphereSize, sphereSize);
    }
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    if (glGetError() != GL_NO_ERROR) fail("OpenGL rendering failed");
    glutSwapBuffers();
}

void timer(int) {
    Vec3 forward = direction(yaw, pitch), right = {sinf(yaw), -cosf(yaw), 0};
    Vec3 movement = add(mul(forward, int(keys['w']) - int(keys['s'])),
                        mul(right, int(keys['d']) - int(keys['a'])));
    movement.z += int(keys['e']) - int(keys['q']);
    float distance = length(movement);
    if (distance > 0) camera = add(camera, mul(movement, 14 * params.dt / distance));
    if (!paused) {
        step_kernel<<<(N + 127) / 128, 128>>>(bodies, nextBodies, velocities, N, camera, 30.f, bullet, params);
        check(cudaGetLastError());
        std::swap(bodies, nextBodies);
        advance(bullet, params.dt);
    }
    glutPostRedisplay();
    glutTimerFunc(10, timer, 0);
}

unsigned char lowercase(unsigned char key) {
    return key >= 'A' && key <= 'Z' ? key + ('a' - 'A') : key;
}

void keyboard(unsigned char key, int, int) {
    key = lowercase(key);
    keys[key] = true;
    if (key == 27) glutLeaveMainLoop();
    if (key == ' ') {
        paused = !paused;
        glutSetWindowTitle(paused ? "Lab 6 | CUDA + OpenGL | PAUSED" : "Lab 6 | CUDA + OpenGL");
    }
    if (key == 'r') reset();
}

void keyboard_up(unsigned char key, int, int) { keys[lowercase(key)] = false; }

void mouse(int button, int state, int x, int y) {
    mouseX = x; mouseY = y;
    if (button == GLUT_RIGHT_BUTTON) looking = state == GLUT_DOWN;
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) bullet = shoot(camera, yaw, pitch);
}

void motion(int x, int y) {
    if (looking) {
        yaw -= (x - mouseX) * .0025f;
        pitch = std::max(-1.55f, std::min(1.55f, pitch - (y - mouseY) * .0025f));
    }
    mouseX = x; mouseY = y;
}

void reshape(int w, int h) {
    width = std::max(w, 1); height = std::max(h, 1);
    glViewport(0, 0, width, height);
}

int main(int argc, char** argv) {
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    setenv("__GL_SHADER_DISK_CACHE", "0", 1);
    setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 0);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 0);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    glutCreateWindow("Lab 6 | CUDA + OpenGL");
    if (glewInit() != GLEW_OK || !GLEW_VERSION_2_1)
        fail("OpenGL 2.1 compatibility context is required");
    glGetError();
    initialize();
    puts("WASD: fly | Q/E: down/up | RMB drag: look | LMB: shoot | Space: pause | R: reset | Esc: exit");
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboard_up);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutCloseFunc(cleanup);
    glutIgnoreKeyRepeat(1);
    glutTimerFunc(10, timer, 0);
    glutMainLoop();
    return 0;
}
