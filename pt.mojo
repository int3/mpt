from algorithm.functional import parallelize
from collections.optional import Optional, OptionalReg
from math import sin, cos, pi, sqrt
from random.random import random_float64


struct Vec(Copyable, Movable):
    var x: Float64
    var y: Float64
    var z: Float64

    fn __init__(out self, x: Float64 = 0, y: Float64 = 0, z: Float64 = 0):
        self.x = x
        self.y = y
        self.z = z

    fn __add__(self, rhs: Vec) -> Self:
        return Vec(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)

    fn __iadd__(mut self, rhs: Vec) -> None:
        self.x += rhs.x
        self.y += rhs.y
        self.z += rhs.z

    fn __sub__(self, rhs: Vec) -> Self:
        return Vec(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)

    fn __mul__(self, rhs: Vec) -> Self:
        return Vec(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)

    fn __mul__(self, scalar: Float64) -> Self:
        return Vec(self.x * scalar, self.y * scalar, self.z * scalar)

    fn __rmul__(self, scalar: Float64) -> Self:
        return self * scalar  # mul is commutative

    fn norm(self) -> Self:
        return self * (1 / sqrt(self.x**2 + self.y**2 + self.z**2))

    fn dot(self, rhs: Vec) -> Float64:
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z

    fn cross(self, rhs: Vec) -> Self:
        return Vec(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )

    fn max(self) -> Float64:
        xy = self.x if self.x > self.y else self.y
        return self.z if self.z > xy else xy


@fieldwise_init
struct Ray(Copyable, Movable):
    var origin: Vec
    var dir: Vec


# Mojo doesn't have enums yet...
alias Material = Int
alias MAT_DIFFUSE: Material = 0
alias MAT_SPECULAR: Material = 1
alias MAT_REFRACT: Material = 2


@fieldwise_init
struct Sphere(Copyable, Movable):
    var radius: Float64
    var pos: Vec
    var emission: Vec
    var color: Vec
    var mat: Material

    fn intersect(self, ray: Ray) -> OptionalReg[Float64]:
        op = self.pos - ray.origin
        b = op.dot(ray.dir)
        det = b**2 - op.dot(op) + self.radius**2
        if det < 0:
            return None
        det = sqrt(det)
        eps = 1e-4
        if b - det > eps:
            return b - det
        elif b + det > eps:
            return b + det
        else:
            return None


alias Scene = List[Sphere]


fn create_scene() -> Scene:
    return [
        Sphere(
            1e5,
            {1e5 + 1, 40.8, 81.6},
            {},
            {0.75, 0.25, 0.25},
            MAT_DIFFUSE,
        ),
        Sphere(
            1e5,
            {-1e5 + 99, 40.8, 81.6},
            {},
            {0.25, 0.25, 0.75},
            MAT_DIFFUSE,
        ),
        Sphere(1e5, {50, 40.8, 1e5}, {}, {0.75, 0.75, 0.75}, MAT_DIFFUSE),
        Sphere(1e5, {50, 40.8, -1e5 + 170}, {}, {}, MAT_DIFFUSE),
        Sphere(1e5, {50, 40.8, -1e5 + 170}, {}, {}, MAT_DIFFUSE),
        Sphere(1e5, {50, 1e5, 81.6}, {}, {0.75, 0.75, 0.75}, MAT_DIFFUSE),
        Sphere(
            1e5,
            {50, -1e5 + 81.6, 81.6},
            {},
            {0.75, 0.75, 0.75},
            MAT_DIFFUSE,
        ),
        Sphere(16.5, {27, 16.5, 47}, {}, {0.999, 0.999, 0.999}, MAT_SPECULAR),
        Sphere(16.5, {73, 16.5, 78}, {}, {0.999, 0.999, 0.999}, MAT_REFRACT),
        Sphere(600, {50, 681.6 - 0.27, 81.6}, {12, 12, 12}, {}, MAT_DIFFUSE),
    ]


fn closest_intersect[
    origin: ImmutableOrigin
](ray: Ray, ref [origin]scene: Scene) -> Optional[
    Tuple[Float64, Pointer[Scene.T, origin]]
]:
    closest = Float64.MAX
    closest_obj: Optional[Pointer[Scene.T, origin]] = None
    for obj in scene:
        dist = obj.intersect(ray)
        if dist is not None and dist.value() < closest:
            closest = dist.value()
            closest_obj = Pointer(to=obj)
    if closest < Float64.MAX:
        return (closest, closest_obj.unsafe_value())
    return None


fn radiance(ray: Ray, owned depth: Int, scene: Scene) -> Vec:
    maybe_intersect = closest_intersect(ray, scene)
    if maybe_intersect is None:
        return Vec()
    dist, obj_ptr = maybe_intersect.unsafe_value()
    ref obj = obj_ptr[]
    hit = ray.origin + ray.dir * dist
    norm = (hit - obj.pos).norm()
    norm_outward = norm if norm.dot(ray.dir) < 0 else norm * (-1)
    f = obj.color

    max_refl = f.max()
    depth += 1
    if depth > 1000:  # avoid stack overflow
        return obj.emission

    if depth > 5:
        if random_float64() < max_refl:
            f = f * (1 / max_refl)
        else:
            return obj.emission

    if obj.mat == MAT_DIFFUSE:
        r1 = 2 * pi * random_float64()
        r2 = random_float64()
        r2s = sqrt(r2)
        w = norm_outward
        u = (Vec(y=1) if abs(w.x) > 0.1 else Vec(x=1)).cross(w).norm()
        v = w.cross(u)
        d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm()
        return obj.emission + f * (radiance({hit, d}, depth, scene))
    elif obj.mat == MAT_SPECULAR:
        return obj.emission + f * (
            radiance(
                {hit, ray.dir - norm * 2 * norm.dot(ray.dir)}, depth, scene
            )
        )
    else:
        refl_ray = Ray(hit, ray.dir - norm * 2 * norm.dot(ray.dir))
        is_into = norm.dot(norm_outward) > 0
        nc = 1
        nt = 1.5
        nnt = nc / nt if is_into else nt / nc
        cos_theta1 = ray.dir.dot(norm_outward)
        cos_theta2_sq = 1 - nnt**2 * (1 - cos_theta1**2)
        if cos_theta2_sq < 0:
            return obj.emission + f * (radiance(refl_ray, depth, scene))
        tdir = (
            ray.dir * nnt
            - norm_outward * (cos_theta1 * nnt + sqrt(cos_theta2_sq))
        ).norm()
        r0 = ((nt - nc) / (nt + nc)) ** 2
        cos_theta_out = -cos_theta1 if is_into else tdir.dot(norm)
        reflectance = r0 + (1 - r0) * (1 - cos_theta_out) ** 5
        transmission = 1 - reflectance
        p = 0.25 + 0.5 * reflectance
        if depth > 2:
            if random_float64() < p:
                rad = radiance(refl_ray, depth, scene) * (reflectance / p)
            else:
                rad = radiance({hit, tdir}, depth, scene) * (
                    transmission / (1 - p)
                )
        else:
            rad = (
                radiance(refl_ray, depth, scene) * reflectance
                + radiance({hit, tdir}, depth, scene) * transmission
            )
        return obj.emission + rad


fn clamp(x: Float64) -> Float64:
    if x < 0:
        return 0
    if x > 1:
        return 1
    return x


fn toIntColor(x: Float64) -> Int:
    return Int(clamp(x) ** (1 / 2.2) * 255 + 0.5)


def main():
    alias w = 1024
    alias h = 768
    image = List(length=w * h, fill=Vec())

    samples = 32
    cam = Ray(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm())
    cx = Vec(x=w * 0.5135 / h)
    cy = cx.cross(cam.dir).norm() * 0.5135
    scene = create_scene()

    fn sample_one_pixel(x: Int, y: Int) -> Vec:
        pixel_radiance = Vec()
        for sy in range(2):
            for sx in range(2):
                subpixel_radiance = Vec()
                for _ in range(samples):
                    r1 = 2 * random_float64()
                    r2 = 2 * random_float64()
                    dx = sqrt(r1) - 1 if r1 < 1 else 1 - sqrt(2 - r1)
                    dy = sqrt(r2) - 1 if r2 < 1 else 1 - sqrt(2 - r2)
                    d = (
                        cx * (((sx + 0.5 + dx) / 2 + x) / w - 0.5)
                        + cy * (((sy + 0.5 + dy) / 2 + y) / h - 0.5)
                        + cam.dir
                    )
                    subpixel_radiance += radiance(
                        {cam.origin + d * 135, d.norm()}, 0, scene
                    ) * (1.0 / samples)
                pixel_radiance += (
                    Vec(
                        clamp(subpixel_radiance.x),
                        clamp(subpixel_radiance.y),
                        clamp(subpixel_radiance.z),
                    )
                    * 0.25
                )
        return pixel_radiance

    @parameter
    fn render_one_row(y: Int):
        for x in range(w):
            i = (h - y - 1) * w + x
            image[i] += sample_one_pixel(x, y)

    parallelize[render_one_row](h)

    with open("image.ppm", "w") as f:
        f.write(String("P3\n{} {}\n255\n").format(w, h))
        for i in range(w * h):
            f.write(
                String("{} {} {} ").format(
                    toIntColor(image[i].x),
                    toIntColor(image[i].y),
                    toIntColor(image[i].z),
                )
            )
