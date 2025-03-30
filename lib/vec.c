#include <import>
#include <math.h>
#include <linmath.h>

// change for vectors and matrices

// vec4f(1, 2, 3, 4)  <- creates a struct literal (vec4f) { args }
// vec4f(obj)         <- creates a struct and calls conversion with_TYPE

#define vec3f_(...) structure(vec3f __VA_OPT__(,) __VA_ARGS__)

#define vec_define_methods(N, T, C) \
    N N##_with_floats(T* f) { \
        N result = {}; \
        if (f) \
            memcpy(&result, f, sizeof(T) * C); \
        else \
            memset(&result, 0, sizeof(T) * C); \
        return result; \
    } \
    N N##_scale(N* a, f32 n) { \
        N res = *a; \
        T* src = a, *f = &res.x; \
        for (int i = 0; i < C; i++) \
            f[i] = src[i] * n; \
        return res; \
    } \
    N N##_add(N* a, N* b) { \
        N res = *a; \
        T* src = a, *src2 = b, *f = &res.x; \
        for (int i = 0; i < C; i++) \
            f[i] = src[i] + src2[i]; \
        return res; \
    } \
    N N##_sub(N* a, N* b) { \
        N res = *a; \
        T*  src = a, *src2 = b, *f = &res.x; \
        for (int i = 0; i < C; i++) \
            f[i] = src[i] - src2[i]; \
        return res; \
    } \
    N N##_mul(N* a, N* b) { \
        N res = *a; \
        T*  src = a, *src2 = b, *f = &res.x; \
        for (int i = 0; i < C; i++) \
            f[i] = src[i] * src2[i]; \
        return res; \
    } \
    N N##_normalize(N* a) { \
        T   len_sq = (T)0.0; \
        for (int i = 0; i < C; i++) \
            len_sq += ((T*)a)[i] * ((T*)a)[i]; \
        T   len    = (T)sqrt((f64)len_sq); \
        if (len > 0) { \
            N res = *a; \
            for (int i = 0; i < C; i++) \
                 ((T*)&res)[i] /= len; \
            return res; \
        } \
        return *a; \
    } \
    N N##_mix(N* a, N* b, f64 f) { \
        N  res = {}; \
        T* fres = &res.x; \
        for (int i = 0; i < C; i++) { \
            T fa  = ((T*)a)[i]; \
            T fb  = ((T*)b)[i]; \
            fres[i] = fa * ((T)1.0 - f) + fb * f; \
        } \
        return res; \
    } \
    T N##_dot(N* a, N* b) { \
        T r = (T)0.0; \
        for (int i = 0; i < C; i++) \
            r += ((T*)a)[i] * ((T*)b)[i]; \
        return r; \
    } \
    T N##_length(N* a) { \
        f64 r = 0.0; \
        for (int i = 0; i < C; i++) \
            r += (f64)((T*)a)[i] * (f64)((T*)a)[i]; \
        return (T)sqrt((f64)r); \
    } \
    string N##_cast_string(N* a) { \
        string res = string(alloc, 1024); \
        append(res, "["); \
        for (int r = 0; r < C; r++) { \
            if (r) append(res, ", "); \
            A_serialize(typeid(T), res, A_ ##T((&a->x)[r])); \
        } \
        append(res, "]"); \
        return res; \
    }

vec_define_methods(vec4f, f32, 4)
vec_define_methods(vec3f, f32, 3)
vec_define_methods(vec2f, f32, 2)

mat4f mat4f_with_floats(f32* f) {
    mat4f a = {};
    if (f)
        memcpy(&a, f, sizeof(f32) * 16);
    else {
        a.m[4 * 0 + 0] = 1.0f;
        a.m[4 * 1 + 1] = 1.0f;
        a.m[4 * 2 + 2] = 1.0f;
        a.m[4 * 3 + 3] = 1.0f;
    }
    return a;
}

quatf quatf_with_floats(f32* f) {
    quatf a = {};
    a.x = f[0];
    a.y = f[1];
    a.z = f[2];
    a.w = f[3]; 
    return a;
}

f32 degrees(f32 rads) { return rads * (180.0f / M_PI); }
f32 radians(f32 degs) { return degs * (M_PI / 180.0f); }

vec4f vec4f_with_vec3f(vec3f* a) {
    return vec4f(a->x, a->y, a->z, 1.0);
}

vec3f vec3f_cross(vec3f* a, vec3f* b) {
    f32 f[3] = {
        a->y * b->z - a->z * b->y,
        a->z * b->x - a->x * b->z,
        a->x * b->y - a->y * b->x
    };
    vec3f v = vec3f((floats)f);
    return v;
}

vec3f vec3f_rand() {
    f32 f[3] = {
        ((float)rand() / RAND_MAX) * 2.0f - 1.0f,
        ((float)rand() / RAND_MAX) * 2.0f - 1.0f,
        ((float)rand() / RAND_MAX) * 2.0f - 1.0f
    };
    return vec3f_normalize(f);
}

/// vec4f treated as axis x/y/z + theta (w) args
quatf quatf_with_vec4f(vec4f* v) {
    quatf q = {}; 
    f32   theta        = v->w;
    f32   half_theta   = theta * 0.5f;
    f32   s_half_theta = sinf(half_theta);
    q.x = v->x * s_half_theta;
    q.y = v->y * s_half_theta;
    q.z = v->z * s_half_theta;
    q.w = cosf(half_theta);
    return q;
}


mat4f mat4f_with_quatf(quatf* q) {
    mat4f mat = {}; 
    /// values are at mat->values[0...15] [ row-major ]
    f32 x = q->x, y = q->y, z = q->z, w = q->w;
    f32 xx = x * x;
    f32 yy = y * y;
    f32 zz = z * z;
    f32 xy = x * y;
    f32 xz = x * z;
    f32 yz = y * z;
    f32 wx = w * x;
    f32 wy = w * y;
    f32 wz = w * z;

    // Fill matrix values in row-major order
    mat.m[0]  = 1.0f - 2.0f * (yy + zz); // Row 1, Col 1
    mat.m[1]  = 2.0f * (xy - wz);        // Row 1, Col 2
    mat.m[2]  = 2.0f * (xz + wy);        // Row 1, Col 3
    mat.m[3]  = 0.0f;                    // Row 1, Col 4

    mat.m[4]  = 2.0f * (xy + wz);        // Row 2, Col 1
    mat.m[5]  = 1.0f - 2.0f * (xx + zz); // Row 2, Col 2
    mat.m[6]  = 2.0f * (yz - wx);        // Row 2, Col 3
    mat.m[7]  = 0.0f;                    // Row 2, Col 4

    mat.m[8]  = 2.0f * (xz - wy);        // Row 3, Col 1
    mat.m[9]  = 2.0f * (yz + wx);        // Row 3, Col 2
    mat.m[10] = 1.0f - 2.0f * (xx + yy); // Row 3, Col 3
    mat.m[11] = 0.0f;                    // Row 3, Col 4

    mat.m[12] = 0.0f;                    // Row 4, Col 1
    mat.m[13] = 0.0f;                    // Row 4, Col 2
    mat.m[14] = 0.0f;                    // Row 4, Col 3
    mat.m[15] = 1.0f;                    // Row 4, Col 4
    return mat;
}
 
none mat4f_set_identity(mat4f* a) {
    memset(a, 0, sizeof(mat4f));
    for (int i = 0; i < 4; i++)
        a->m[4 * i + i] = 1.0f;
}

mat4f mat4f_mul(mat4f* a, mat4f* b) {
    mat4f res = {};
    int cols_a = 4;
    int cols_b = 4;
    // Perform multiplication
    for (i64 i = 0; i < 4; ++i) {
        for (i64 j = 0; j < 4; ++j) {
            res.m[i * cols_b + j] = 0; // Initialize element
            for (i64 k = 0; k < cols_a; ++k) {
                res.m[i * cols_b + j] += a->m[i * cols_a + k] * b->m[k * cols_b + j];
            }
        }
    }
    return res;
}

vec4f mat4f_mul_v4(mat4f* a, vec4f* b) {
    vec4f res  = {};
    for (i64 i = 0; i < 4; ++i)
        for (i64 j = 0; j < 4; ++j)
            (&res.x)[i] += a->m[i * 4 + j] * (&b->x)[j];
    return res;
}

mat4f mat4f_scale(mat4f* a, vec3f* f) {
    u32 size = 4;
    mat4f r = *a;
    for (u32 i = 0; i < 3; ++i)
        r.m[i * (size + 1)] = (&f->x)[i];
    return r;
}

// any 'shape' in A-type model applies on top of vmember_count
mat4f mat4f_translate(mat4f* a, vec3f* offsets) {
    mat4f tr = mat4f_ident();
    tr.m[12] = offsets->x; // Column-major: m[12] = (3,0)
    tr.m[13] = offsets->y; // Column-major: m[13] = (3,1)
    tr.m[14] = offsets->z; // Column-major: m[14] = (3,2)
    return mat4f_mul(a, &tr);
}

mat4f mat4f_look_at(vec3f* eye, vec3f* target, vec3f* up) {

    mat4f res = {};
    mat4x4_look_at(&res, *(vec3*)eye, *(vec3*)target, *(vec3*)up);
    return res;

    //LINMATH_H_FUNC void mat4x4_look_at(mat4x4 m, vec3 const eye, vec3 const center, vec3 const up)

    vec3f diff    = vec3f_sub(target, eye);
    vec3f forward = vec3f_normalize(&diff); // Z-axis (points away from target)
    vec3f rcross  = vec3f_cross(up, &forward);
    vec3f right   = vec3f_normalize(&rcross);  // X-axis
    vec3f new_up  = vec3f_cross(&forward, &right); // Y-axis (orthogonalized)

    // Construct the view matrix
    mat4f r = {};
    r.m[ 0] = right.x;  r.m[ 1] = new_up.x;  r.m[ 2] = forward.x;  r.m[ 3] = 0.0f;
    r.m[ 4] = right.y;  r.m[ 5] = new_up.y;  r.m[ 6] = forward.y;  r.m[ 7] = 0.0f;
    r.m[ 8] = right.z;  r.m[ 9] = new_up.z;  r.m[10] = forward.z;  r.m[11] = 0.0f;
    r.m[12] = -vec3f_dot(&right, eye);
    r.m[13] = -vec3f_dot(&new_up, eye);
    r.m[14] = -vec3f_dot(&forward, eye);
    r.m[15] = 1.0f;
    return r;
}

mat4f mat4f_ortho(f32 left, f32 right, f32 bottom, f32 top, f32 near, f32 far) {
    f32 rl = right - left;
    f32 tb = top   - bottom;
    f32 fn = far   - near;
    mat4f res = {};
    // Construct the orthographic projection matrix
    res.m[ 0] = 2.0f / rl;  res.m[ 1] = 0.0f;       res.m[ 2] = 0.0f;        res.m[ 3] = 0.0f;
    res.m[ 4] = 0.0f;       res.m[ 5] = 2.0f / tb;  res.m[ 6] = 0.0f;        res.m[ 7] = 0.0f;
    res.m[ 8] = 0.0f;       res.m[ 9] = 0.0f;       res.m[10] = -2.0f / fn;  res.m[11] = 0.0f;
    res.m[12] = -(right + left)   / rl;
    res.m[13] = -(top   + bottom) / tb;
    res.m[14] = -(far   + near)   / fn;
    res.m[15] = 1.0f;
    return res;
}

f32 mat4f_determinant(mat4f* mat) {
    f32 *m = mat->m; // Access matrix elements directly
    f32 det = 
        m[0] * (m[5] * (m[10] * m[15] - m[11] * m[14]) -
                m[6] * (m[9] * m[15] - m[11] * m[13]) +
                m[7] * (m[9] * m[14] - m[10] * m[13])) -
        m[1] * (m[4] * (m[10] * m[15] - m[11] * m[14]) -
                m[6] * (m[8] * m[15] - m[11] * m[12]) +
                m[7] * (m[8] * m[14] - m[10] * m[12])) +
        m[2] * (m[4] * (m[9] * m[15] - m[11] * m[13]) -
                m[5] * (m[8] * m[15] - m[11] * m[12]) +
                m[7] * (m[8] * m[13] - m[9] * m[12])) -
        m[3] * (m[4] * (m[9] * m[14] - m[10] * m[13]) -
                m[5] * (m[8] * m[14] - m[10] * m[12]) +
                m[6] * (m[8] * m[13] - m[9] * m[12]));

    return det;
}

mat4f mat4f_transpose(mat4f* mat) {
    mat4f r = {};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            r.m[i * 4 + j] = mat->m[j * 4 + i];
    }
    return r;
}

f32 determinant_3x3(f32 a, f32 b, f32 c, f32 d, f32 e, f32 f, f32 g, f32 h, f32 i) {
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

// mutable
mat4f mat4f_adjugate(mat4f* mat) {
    f32 *m = mat->m;
    mat4f r = {};

    r.m[0]  =  determinant_3x3(m[5],  m[6],  m[7],  m[9],  m[10], m[11], m[13], m[14], m[15]);
    r.m[1]  = -determinant_3x3(m[1],  m[2],  m[3],  m[9],  m[10], m[11], m[13], m[14], m[15]);
    r.m[2]  =  determinant_3x3(m[1],  m[2],  m[3],  m[5],  m[6],  m[7],  m[13], m[14], m[15]);
    r.m[3]  = -determinant_3x3(m[1],  m[2],  m[3],  m[5],  m[6],  m[7],  m[9],  m[10], m[11]);

    r.m[4]  = -determinant_3x3(m[4],  m[6],  m[7],  m[8],  m[10], m[11], m[12], m[14], m[15]);
    r.m[5]  =  determinant_3x3(m[0],  m[2],  m[3],  m[8],  m[10], m[11], m[12], m[14], m[15]);
    r.m[6]  = -determinant_3x3(m[0],  m[2],  m[3],  m[4],  m[6],  m[7],  m[12], m[14], m[15]);
    r.m[7]  =  determinant_3x3(m[0],  m[2],  m[3],  m[4],  m[6],  m[7],  m[8],  m[10], m[11]);

    r.m[8]  =  determinant_3x3(m[4],  m[5],  m[7],  m[8],  m[9],  m[11], m[12], m[13], m[15]);
    r.m[9]  = -determinant_3x3(m[0],  m[1],  m[3],  m[8],  m[9],  m[11], m[12], m[13], m[15]);
    r.m[10] =  determinant_3x3(m[0],  m[1],  m[3],  m[4],  m[5],  m[7],  m[12], m[13], m[15]);
    r.m[11] = -determinant_3x3(m[0],  m[1],  m[3],  m[4],  m[5],  m[7],  m[8],  m[9],  m[11]);

    r.m[12] = -determinant_3x3(m[4],  m[5],  m[6],  m[8],  m[9],  m[10], m[12], m[13], m[14]);
    r.m[13] =  determinant_3x3(m[0],  m[1],  m[2],  m[8],  m[9],  m[10], m[12], m[13], m[14]);
    r.m[14] = -determinant_3x3(m[0],  m[1],  m[2],  m[4],  m[5],  m[6],  m[12], m[13], m[14]);
    r.m[15] =  determinant_3x3(m[0],  m[1],  m[2],  m[4],  m[5],  m[6],  m[8],  m[9],  m[10]);

    // Adjugate is the transpose of the cofactor matrix
    mat4f_transpose(&r);
    return r;
}

mat4f mat4f_inverse(mat4f* mat) {
    // Compute the determinant
    f32 det = mat4f_determinant(mat);
    mat4f res = {};
    if (fabs(det) < 1e-6f) {
        fault("Matrix is singular (non-invertible)");
    }

    // Compute the ifnverse
    f32 inv_det = 1.0f / det;
    mat4f adj = mat4f_adjugate(mat); // Compute adjugate
    for (int i = 0; i < 16; i++) {
        res.m[i] = adj.m[i] * inv_det;
    }
    return res;
}

mat4f mat4f_ident() {
    return mat4f((floats)null);
}

mat4f mat4f_perspective(f32 y_fov, f32 aspect, f32 n, f32 f) {
	float const ifov = 1.f / tanf(y_fov / 2.f);
    f32 m[4][4];
    memset(m, 0, sizeof(m));
	m[0][0] = ifov / aspect;
	m[0][1] = 0.f;
	m[0][2] = 0.f;
	m[0][3] = 0.f;

	m[1][0] = 0.f;
	m[1][1] = ifov;
	m[1][2] = 0.f;
	m[1][3] = 0.f;

	m[2][0] = 0.f;
	m[2][1] = 0.f;
	m[2][2] = -((f + n) / (f - n));
	m[2][3] = -1.f;

	m[3][0] = 0.f;
	m[3][1] = 0.f;
	m[3][2] = -((2.f * f * n) / (f - n));
	m[3][3] = 0.f;
    mat4f res;
    memcpy(&res, m, sizeof(m));
    return res;
}


mat4f mat4f_rotate(mat4f* mat, quatf* q) {
    mat4f res = {};

    // quaternion rotation
    f32 x = q->x, y = q->y, z = q->z, w = q->w;
    f32 xx = x * x, yy = y * y, zz = z * z;
    f32 xy = x * y, xz = x * z, yz = y * z;
    f32 wx = w * x, wy = w * y, wz = w * z;

    res.m[0]  = 1.0f - 2.0f * (yy + zz);
    res.m[1]  = 2.0f * (xy - wz);
    res.m[2]  = 2.0f * (xz + wy);
    res.m[3]  = 0.0f;

    res.m[4]  = 2.0f * (xy + wz);
    res.m[5]  = 1.0f - 2.0f * (xx + zz);
    res.m[6]  = 2.0f * (yz - wx);
    res.m[7]  = 0.0f;

    res.m[8]  = 2.0f * (xz - wy);
    res.m[9]  = 2.0f * (yz + wx);
    res.m[10] = 1.0f - 2.0f * (xx + yy);
    res.m[11] = 0.0f;

    res.m[12] = 0.0f;
    res.m[13] = 0.0f;
    res.m[14] = 0.0f;
    res.m[15] = 1.0f;
    return mat4f_mul(mat, &res);
}

string mat4f_cast_string(mat4f* a) {
    bool  once = false;
    string res = string(alloc, 1024);
    append(res, "[");
    for (int i = 0; i < 4 * 4; i++) {
        if (i) append(res, ", ");
        A_serialize(typeid(f32), res, A_f32(a->m[i]));
    }
    append(res, "]");
    return res;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvarargs"

/// replaces uses of 'sampler'
#define vector_impl(T, ARG_T) \
vector_##T vector_##T##_new(shape vshape, ...) { \
    va_list args; \
    va_start(args, vshape); \
    vector_##T result = vector_##T(shape, vshape); \
    T* T##_data = data(result); \
    for (int i = 0, count = total(vshape); i < count; i++) { \
        T##_data[i] = (T)va_arg(args, ARG_T); \
    } \
    return result; \
}

#pragma GCC diagnostic pop

vector_impl(i8,    i64)
vector_impl(i64,   i64)
vector_impl(f32,   f64)
vector_impl(f64,   f64)
vector_impl(rgb8,  rgb8)
vector_impl(rgbf,  rgbf)
vector_impl(rgba8, rgba8)
vector_impl(rgbaf, rgbaf)

rgbaf rgbaf_with_vec4f(vec4f* v4) {
    return *(rgbaf*)v4;
}

define_struct(rgb8, u8)
define_vector(rgb8, u8, 3)

define_struct(rgbf, f32)
define_vector(rgbf, f32, 3)

define_struct(rgba8, u8)
define_vector(rgba8, u8, 4)

define_struct(rgbaf, f32)
define_vector(rgbaf, f32, 4)

define_struct(quatf, f32)
define_vector(quatf, f32, 4)

define_struct(vec4f, f32)
define_vector(vec4f, f32, 4)

define_struct(vec3f, f32)
define_vector(vec3f, f32, 3)

define_struct(vec2f, f32)
define_vector(vec2f, f32, 2)

define_struct(mat4f, f32)
define_vector(mat4f, f32, 16)

define_meta(vector_rgbf,  vector, rgbf)
define_meta(vector_rgb8,  vector, rgb8)
define_meta(vector_rgbaf, vector, rgbaf)
define_meta(vector_rgba8, vector, rgba8)
define_meta(vector_i8,    vector, i8)
define_meta(vector_i64,   vector, i64)
define_meta(vector_f32,   vector, f32)
define_meta(vector_f64,   vector, f64)

/// vector class works with structs in meta
define_meta(vector_mat4f,      vector, mat4f)

