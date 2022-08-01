#include <unordered_set>
#include <pugixml/pugixml.hpp>

#include <misc/Exception.h>
#include <psdr/core/bitmap.h>
#include <psdr/core/ray.h>
#include <psdr/core/frame.h>
#include <psdr/core/transform.h>
#include <psdr/core/sampler.h>
#include <psdr/bsdf/diffuse.h>
#include <psdr/bsdf/roughconductor.h>
#include <psdr/bsdf/hetersub.h>
#include <psdr/bsdf/layersub.h>
#include <psdr/bsdf/microfacet.h>
#include <psdr/emitter/area.h>
#include <psdr/emitter/point.h>
#include <psdr/emitter/envmap.h>
#include <psdr/sensor/perspective.h>
#include <psdr/shape/mesh.h>

#include <psdr/scene/scene.h>
#include <psdr/scene/scene_loader.h>

namespace psdr
{

template <int length>
Array<float, length> parse_vector(const char *str, bool allow_empty = false) {
    Array<float, length> result;
    int tot = 0;

    int i = 0;
    for ( ; ; ) {
        while ( str[i] && strchr(", ", str[i]) ) ++i;
        if ( !str[i] ) break;

        int j = i + 1;
        while ( str[j] && strchr(", ", str[j]) == nullptr ) ++j;

        PSDR_ASSERT(tot < length);
        result[tot++] = static_cast<float>(atof(str + i));

        i = j;
    }

    if ( tot < length ) {
        if ( allow_empty ) {
            float value = tot ? result[tot - 1] : 0.0f;
            std::fill(result.data() + tot, result.data() + length, value);
        } else {
            PSDR_ASSERT_MSG(false, std::string("Vector too short: [") + str + "]");
        }
    }

    return result;
}


inline static std::string parse_bitmap(const pugi::xml_node &node) {
    const char *texture_type = node.attribute("type").value();
    PSDR_ASSERT_MSG(strcmp(texture_type, "bitmap") == 0, std::string("Unsupported texture type: ") + texture_type);

    const pugi::xml_node &fn_node = node.child("string");
    const char *tmp = fn_node.attribute("name").value(), *file_name = fn_node.attribute("value").value();
    PSDR_ASSERT_MSG(strcmp(tmp, "filename") == 0 && file_name, "Failed to retrieve bitmap filename");
    return std::string(file_name);
}


inline static pugi::xml_node find_child_by_name(const pugi::xml_node &parent,
                                                const std::unordered_set<std::string> &names,
                                                bool allow_empty = false) {
    PSDR_ASSERT(!names.empty());
    pugi::xml_node result = parent.find_child(
        [&](pugi::xml_node node) {
            return names.find(node.attribute("name").value()) != names.end();
        }
    );
    PSDR_ASSERT_MSG(allow_empty || result, std::string("Missing child node: ") + *names.begin());
    return result;
}


static ScalarMatrix4f load_transform(const pugi::xml_node &parent) {
    ScalarMatrix4f result = identity<ScalarMatrix4f>();

    if ( parent ) {
        const char *node_name = parent.attribute("name").value();
        PSDR_ASSERT_MSG(strcmp(node_name, "to_world") == 0 || strcmp(node_name, "toWorld") == 0,
                        std::string("Invalid transformation name: ") + node_name);

        for ( auto node = parent.first_child(); node; node = node.next_sibling() ) {
            if ( strcmp(node.name(), "translate") == 0 ) {
                ScalarVector3f offset(
                    node.attribute("x").as_float(0.0f),
                    node.attribute("y").as_float(0.0f),
                    node.attribute("z").as_float(0.0f)
                );
                result = transform::translate(offset)*result;
            } else if ( strcmp(node.name(), "rotate") == 0 ) {
                ScalarVector3f axis(
                    node.attribute("x").as_float(),
                    node.attribute("y").as_float(),
                    node.attribute("z").as_float()
                );
                float angle = node.attribute("angle").as_float();
                result = transform::rotate(axis, angle)*result;
            } else if ( strcmp(node.name(), "scale") == 0 ) {
                ScalarVector3f scl(
                    node.attribute("x").as_float(1.0f),
                    node.attribute("y").as_float(1.0f),
                    node.attribute("z").as_float(1.0f)
                );
                result = transform::scale(scl)*result;
            } else if ( strcmp(node.name(), "look_at") == 0 || strcmp(node.name(), "lookAt") == 0 ||
                        strcmp(node.name(), "lookat") == 0 ) {
                ScalarVector3f origin = parse_vector<3>(node.attribute("origin").value()),
                               target = parse_vector<3>(node.attribute("target").value()),
                               up = parse_vector<3>(node.attribute("up").value());
                result = transform::look_at(origin, target, up)*result;
            } else if ( strcmp(node.name(), "matrix") == 0 ) {
                Array<float, 16> buf = parse_vector<16>(node.attribute("value").value());
                result = transpose(load<ScalarMatrix4f>(buf.data()))*result;
            } else {
                PSDR_ASSERT_MSG(false, std::string("Unsupported transformation: ") + node.name());
            }
        }
    }

    return result;
}


static int load_sampler(const pugi::xml_node &node) {
    return node.child("integer").attribute("value").as_int();
}


static std::pair<int, int> load_film(const pugi::xml_node &node) {
    pugi::xml_node child;

    child = find_child_by_name(node, {"width"});
    int width = child.attribute("value").as_int();
    child = find_child_by_name(node, {"height"});
    int height = child.attribute("value").as_int();

    return { width, height };
}

static ScalarVector3f load_rgb(const pugi::xml_node &node) {
    if ( strcmp(node.name(), "float") == 0 ) {
        return ScalarVector3f(node.attribute("value").as_float());
    } else if ( strcmp(node.name(), "rgb") == 0 ) {
        return parse_vector<3>(node.attribute("value").value(), true);
    } else {
        // std::cout<<"node name"<<std::node.name();
        PSDR_ASSERT_MSG(false, std::string("Unsupported RGB type: ") + node.name());
    }
}


template <int nchannels>
void load_texture(const pugi::xml_node &node, Bitmap<nchannels> &bitmap) {
    //std::cout<<"loading..."<<node.name()<<std::endl;
    if ( strcmp(node.name(), "texture") == 0) {
        bitmap.load_openexr(parse_bitmap(node).c_str());
    } else {
        if constexpr ( nchannels == 1 ) {
            PSDR_ASSERT(node.attribute("value").value());
            bitmap.fill(node.attribute("value").as_float());
        } else {
            bitmap.fill(load_rgb(node));
        }
    }
}


void SceneLoader::load_from_file(const char *file_name, Scene &scene) {
    pugi::xml_document doc;
    PSDR_ASSERT_MSG(doc.load_file(file_name), "XML parsing failed");
    load_scene(doc, scene);
}


void SceneLoader::load_from_string(const char *scene_xml, Scene &scene) {
    pugi::xml_document doc;
    PSDR_ASSERT_MSG(doc.load_string(scene_xml), "XML parsing failed");
    load_scene(doc, scene);
}


template <typename T>
void build_param_map(Scene::ParamMap &param_map, const std::vector<T*> arr, const char *name) {
    for ( size_t i = 0; i < arr.size(); ++i ) {
        const T *obj = arr[i];

        std::stringstream oss1;
        oss1 << name << "[" << i << "]";
        param_map.insert(Scene::ParamMap::value_type(oss1.str(), *obj));

        if ( obj->m_id != "" ) {
            std::stringstream oss2;
            oss2 << name << "[id=" << obj->m_id << "]";

            bool is_new;
            std::tie(std::ignore, is_new) = param_map.insert(Scene::ParamMap::value_type(oss2.str(), *obj));
            PSDR_ASSERT_MSG(is_new, std::string("Duplicate id: ") + obj->m_id);
        }
    }
}


void SceneLoader::load_scene(const pugi::xml_document &doc, Scene &scene) {
    PSDR_ASSERT_MSG(!scene.m_loaded, "Scene already loaded!");
    
    const pugi::xml_node &root = doc.child("scene");

    // Load sensors
    //std::cout<<"Loading sensor"<<::std::endl;
    for ( auto node = root.child("sensor"); node; node = node.next_sibling("sensor") ) {
        // std::cout<<node.attribute("id").value();
        
        load_sensor(node, scene);
    }

    // Load BSDFs
    //std::cout<<"Loading material"<<::std::endl;
    for ( auto node = root.child("bsdf"); node; node = node.next_sibling("bsdf") ) {
        load_bsdf(node, scene);
    }

    //std::cout<<"Loading emitter"<<::std::endl;
    // Load (env) emitter
    for ( auto node = root.child("emitter"); node; node = node.next_sibling("emitter") ) {
        load_emitter(node, scene);
    }

    //std::cout<<"Loading shape"<<::std::endl;
    // Load shapes
    for ( auto node = root.child("shape"); node; node = node.next_sibling("shape") ) {
        load_shape(node, scene);
    }

    // Build the parameter map
   // std::cout<<"build param map"<<::std::endl;
    build_param_map<Mesh   >(scene.m_param_map, scene.m_meshes  , "Mesh"   );
    build_param_map<Emitter>(scene.m_param_map, scene.m_emitters, "Emitter");
    build_param_map<Sensor >(scene.m_param_map, scene.m_sensors , "Sensor" );

    scene.m_num_sensors = static_cast<int>(scene.m_sensors.size());
    scene.m_num_meshes = static_cast<int>(scene.m_meshes.size());

    scene.m_loaded = true;
    //std::cout<<"scene loaded"<<::std::endl;
}


void SceneLoader::load_sensor(const pugi::xml_node &node, Scene &scene) {
    const char *sensor_type = node.attribute("type").value();

    const pugi::xml_node &film_node = node.child("film");
    const pugi::xml_node &sampler_node = node.child("sampler");
    if ( scene.m_sensors.empty() ) {
        PSDR_ASSERT_MSG(film_node, "Missing film node");
        PSDR_ASSERT_MSG(sampler_node, "Missing sampler node");

        RenderOption &opts = scene.m_opts;
        std::tie(opts.width, opts.height) = load_film(film_node);
        opts.cropwidth = opts.width;
        opts.cropheight = opts.height;
        opts.cropoffset_x = 0;
        opts.cropoffset_y = 0;
        opts.spp = opts.sppe = opts.sppse = opts.sppsce = load_sampler(sampler_node);
    } else {
        PSDR_ASSERT_MSG(!film_node, "Duplicate film node");
        PSDR_ASSERT_MSG(!sampler_node, "Duplicate sampler node");
    }

    if ( strcmp(sensor_type, "perspective") == 0 ) {
        // Perspective camera
        ScalarMatrix4f to_world = load_transform(node.child("transform"));

        float fov_x = find_child_by_name(node, {"fov"}).attribute("value").as_float();

        const pugi::xml_node &fov_axis_node = find_child_by_name(node, {"fov_axis", "fovAxis"}, true);
        if ( fov_axis_node ) {
            const char *fov_axis = fov_axis_node.attribute("value").value();
            if ( strcmp(fov_axis, "x") ) {
                PSDR_ASSERT_MSG(false, std::string("Unsupported fov-axis: ") + fov_axis);
            }
        }

        const pugi::xml_node &near_node = find_child_by_name(node, {"near_clip", "nearClip"}, true);
        float near = ( near_node ? near_node.attribute("value").as_float(0.1f) : 0.1f );

        const pugi::xml_node &far_node = find_child_by_name(node, {"far_clip", "farClip"}, true);
        float far = ( far_node ? far_node.attribute("value").as_float(1e4f) : 1e4f );

        Sensor *sensor = new PerspectiveCamera(fov_x, near, far);
        sensor->m_to_world = Matrix4fD(to_world);
        scene.m_sensors.push_back(sensor);
    } else {
        PSDR_ASSERT_MSG(false, std::string("Unsupported sensor: ") + sensor_type);
    }
}


void SceneLoader::load_emitter(const pugi::xml_node &node, Scene &scene) {
    const char *emitter_type = node.attribute("type").value();

    if ( strcmp(emitter_type, "envmap") == 0 ) {
        // Environment map
        PSDR_ASSERT_MSG(scene.m_emitter_env == nullptr, "A scene is only allowed to have one envmap!");

        const pugi::xml_node &fn_node = node.child("string");
        const char *tmp = fn_node.attribute("name").value(), *file_name = fn_node.attribute("value").value();
        PSDR_ASSERT_MSG(strcmp(tmp, "filename") == 0 && file_name, "Failed to retrieve bitmap filename");

        const pugi::xml_node &scale_node = find_child_by_name(node, { "scale" }, true);
        float scale = (scale_node ? scale_node.attribute("value").as_float(1.f) : 1.f);

        ScalarMatrix4f to_world = load_transform(node.child("transform"));

        EnvironmentMap *emitter = new EnvironmentMap(file_name);
        emitter->m_scale = scale;
        emitter->m_to_world_raw = Matrix4fD(to_world);
        scene.m_emitters.push_back(emitter);
        scene.m_emitter_env = emitter;

    }else if(strcmp(emitter_type, "point") == 0 ){
        PointLight *emitter = new PointLight(load_rgb(find_child_by_name(node, {"power"})), Matrix4fD(load_transform(find_child_by_name(node, {"to_world"}))));
        scene.m_emitters.push_back(emitter);
    }else {
        PSDR_ASSERT_MSG(false, std::string("Unsupported emitter: ") + emitter_type);
    }
}


void SceneLoader::load_bsdf(const pugi::xml_node &node, Scene &scene) {
    const char *bsdf_id = node.attribute("id").value();
    PSDR_ASSERT_MSG(bsdf_id && strcmp(bsdf_id, ""), "BSDF must have an id");

    BSDF* bsdf = nullptr;
    const char *bsdf_type = node.attribute("type").value();
    if ( strcmp(bsdf_type, "diffuse") == 0 ) {
        // Diffuse BSDF

        pugi::xml_node refl_node = find_child_by_name(node, {"reflectance"});

        Diffuse *b = new Diffuse();
        load_texture(refl_node, b->m_reflectance);

        bsdf = b;
    } else if ( strcmp(bsdf_type, "roughconductor") == 0 ){
        // roughconductor BSDF
        pugi::xml_node alpha = find_child_by_name(node, {"alpha"});
        pugi::xml_node eta = find_child_by_name(node, {"eta"});
        pugi::xml_node k = find_child_by_name(node, {"k"});

        RoughConductor *b = new RoughConductor();
        load_texture(alpha, b->m_alpha_u);
        load_texture(alpha, b->m_alpha_v);
        load_texture(eta, b->m_eta);
        load_texture(k, b->m_k);
        bsdf = b;
    } 
    // else if( strcmp(bsdf_type, "subsurface") == 0){
    //     // subsurface material
    //     pugi::xml_node alpha = find_child_by_name(node, {"alpha"});
    //     pugi::xml_node eta = find_child_by_name(node, {"eta"});
    //     pugi::xml_node reflectance = find_child_by_name(node, {"reflectance"});
    //     pugi::xml_node albedo = find_child_by_name(node, {"albedo"});
    //     pugi::xml_node sigma_t = find_child_by_name(node, {"sigma_t"});
    //     pugi::xml_node specular_reflectance = find_child_by_name(node, {"specular_reflectance"});
    //     pugi::xml_node specular_prob = find_child_by_name(node, {"specular_prob"});
        
    //     Subsurface *b = new Subsurface();
    //     load_texture(alpha, b->m_alpha_u);
    //     load_texture(alpha, b->m_alpha_v);
    //     load_texture(eta, b->m_eta);
    //     load_texture(specular_reflectance, b->m_specular_reflectance);
    //     load_texture(sigma_t, b->m_sigma_t);
    //     load_texture(albedo, b->m_albedo);
    //     load_texture(reflectance, b->m_reflectance);
    //     load_texture(specular_prob, b->m_specular_prob);
    //     bsdf = b;
    // }
    else if(strcmp(bsdf_type, "hetersub") == 0){
        pugi::xml_node alpha = find_child_by_name(node, {"alpha"});
        pugi::xml_node eta = find_child_by_name(node, {"eta"});
        pugi::xml_node albedo = find_child_by_name(node, {"albedo"});
        pugi::xml_node sigma_tr = find_child_by_name(node, {"sigma_t"});
        pugi::xml_node specular_reflectance = find_child_by_name(node, {"specular_reflectance"});
        
        HeterSub *b = new HeterSub();
        load_texture(alpha, b->m_alpha_u);
        load_texture(alpha, b->m_alpha_v);
        load_texture(eta, b->m_eta);
        load_texture(specular_reflectance, b->m_specular_reflectance);
        load_texture(sigma_tr, b->m_sigma_t);
        load_texture(albedo, b->m_albedo);
        bsdf = b;
    }else if(strcmp(bsdf_type, "microfacet") == 0){
        pugi::xml_node alpha = find_child_by_name(node, {"alpha"});
        pugi::xml_node reflectance = find_child_by_name(node, {"reflectance"});
        pugi::xml_node specular_reflectance = find_child_by_name(node, {"specular_reflectance"});

        Microfacet *b = new Microfacet();
        load_texture(alpha, b->m_roughness);
        load_texture(specular_reflectance, b->m_specularReflectance);
        load_texture(reflectance, b->m_diffuseReflectance);
        bsdf = b;
    }
    // else if(strcmp(bsdf_type, "layersub") == 0){
    //     pugi::xml_node alpha = find_child_by_name(node, {"alpha"});
    //     pugi::xml_node eta = find_child_by_name(node, {"eta"});
    //     pugi::xml_node reflectance = find_child_by_name(node, {"reflectance"});
    //     pugi::xml_node albedo = find_child_by_name(node, {"albedo"});
    //     pugi::xml_node sigma_tr = find_child_by_name(node, {"sigma_t"});
    //     pugi::xml_node specular_reflectance = find_child_by_name(node, {"specular_reflectance"});
    //     pugi::xml_node specular_prob = find_child_by_name(node, {"specular_prob"});
    //     pugi::xml_node layers = find_child_by_name(node, {"layer_count"});
    //     pugi::xml_node maxdepth = find_child_by_name(node, {"max_depth"});
    //     pugi::xml_node layerdepthnode = find_child_by_name(node, {"layer_depth"});

    //     int layercount = layers.attribute("value").as_int();
    //     float layermaxdepth = maxdepth.attribute("value").as_float();
    //     float layerdepth = layerdepthnode.attribute("value").as_float();
        
    //     LayerSub *b = new LayerSub();
    //     load_texture(alpha, b->m_alpha_u);
    //     load_texture(alpha, b->m_alpha_v);
    //     load_texture(eta, b->m_eta);
    //     load_texture(specular_reflectance, b->m_specular_reflectance);
    //     load_texture(sigma_tr, b->m_sigma_t);
    //     load_texture(albedo, b->m_albedo);
    //     load_texture(reflectance, b->m_reflectance);
    //     load_texture(specular_prob, b->m_specular_prob);
    //     b->m_layerDepth = layerdepth;
    //     b->m_layers = layercount;
    //     b->m_maxDepth = layermaxdepth;
    //     bsdf = b;
        
    // }
    else{
        PSDR_ASSERT_MSG(false, std::string("Unsupported BSDF: ") + bsdf_type);
    }

    bsdf->m_id = bsdf_id;
    scene.m_bsdfs.push_back(bsdf);

    Scene::ParamMap &param_map = scene.m_param_map;

    std::stringstream oss1, oss2;
    oss1 << "BSDF[" << scene.m_bsdfs.size() - 1 << "]";
    oss2 << "BSDF[id=" << bsdf_id << "]";
    param_map.insert(Scene::ParamMap::value_type(oss1.str(), *bsdf));

    bool is_new;
    std::tie(std::ignore, is_new) = param_map.insert(Scene::ParamMap::value_type(oss2.str(), *bsdf));
    PSDR_ASSERT_MSG(is_new, std::string("Duplicate BSDF id: ") + bsdf_id);
}


void SceneLoader::load_shape(const pugi::xml_node &node, Scene &scene) {
    const char *mesh_id = node.attribute("id").value();

    const char *shape_type = node.attribute("type").value();
    Mesh *mesh = nullptr;
    if ( strcmp(shape_type, "obj") == 0 ) {
        const pugi::xml_node &name_node = node.child("string");
        PSDR_ASSERT(strcmp(name_node.attribute("name").value(), "filename") == 0);

        // Load mesh
        const char *file_name = name_node.attribute("value").value();
        mesh = new Mesh();
        mesh->load(file_name);
    } else {
        PSDR_ASSERT_MSG(false, std::string("Unsupported shape: ") + shape_type);
    }

    // duplicate mesh layer
    const pugi::xml_node &int_node = node.child("int");
    PSDR_ASSERT(strcmp(int_node.attribute("name").value(), "layer_count") == 0);
    const pugi::xml_node &float_node = node.child("float");
    PSDR_ASSERT(strcmp(float_node.attribute("name").value(), "layer_depth") == 0);
    const int layer_count = std::atoi(int_node.attribute("value").value());
    const float layer_depth = std::atof(float_node.attribute("value").value());
    //
    //std::cout<<" triangle uv indices "<<mesh->m_face_uv_indices<<std::endl;
    if (layer_count > 0)
        mesh->m_layer_count = 0;

    // generate layer meshes
    std::vector<Mesh *> layer_instances;
    layer_instances.resize(layer_count);
    for( int i = 0 ; i < layer_count ; i++){
        layer_instances[i] = new Mesh();
        layer_instances[i]->instance(mesh, layer_depth * (i + 1));
        layer_instances[i]->m_layer_count = i + 1;
        // std::cout<<"uvs "<<layer_instances[i]->m_vertex_uv<<std::endl;
       // std::cout<<" triangle uv indices "<<layer_instances[i]->m_face_uv_indices<<std::endl;
    }


    // Set BSDF
    const pugi::xml_node &ref_node = node.child("ref");
    PSDR_ASSERT_MSG(ref_node, std::string("Missing BSDF reference"));

    const char *bsdf_id = ref_node.attribute("id").value();
    PSDR_ASSERT(bsdf_id);

    std::stringstream oss;
    oss << "BSDF[id=" << bsdf_id << "]";
    auto bsdf_info = scene.m_param_map.find(oss.str());
    PSDR_ASSERT_MSG(bsdf_info != scene.m_param_map.end(), std::string("Unknown BSDF id: ") + bsdf_id);
    mesh->m_bsdf = dynamic_cast<const BSDF*>(&bsdf_info->second);

    // set values for instances
    for (int i = 0 ; i < layer_count ; i++){
        layer_instances[i]->m_bsdf = mesh->m_bsdf;
    }

    PSDR_ASSERT_MSG(!node.child("bsdf"), "BSDFs declared under shapes are not supported.");

    // Handle face normals
    bool use_face_normals = false;
    const pugi::xml_node &fn_node = find_child_by_name(node, {"face_normals", "faceNormals"}, true);
    if ( fn_node )
        use_face_normals = (strcmp(fn_node.attribute("value").value(), "true") == 0);

    if ( mesh_id ) mesh->m_id = mesh_id;
    mesh->m_use_face_normals = use_face_normals;

    // set normal information for instances
    for (int i = 0 ; i < layer_count ; i++){
        layer_instances[i]->m_id = mesh->m_id + std::to_string(i);
        layer_instances[i]->m_use_face_normals = use_face_normals;
    }

    // Set emitter
    const pugi::xml_node &emitter_node = node.child("emitter");
    if ( emitter_node ) {
        const char *emitter_type = emitter_node.attribute("type").value();
        PSDR_ASSERT_MSG(strcmp(emitter_type, "area") == 0, std::string("Unsupported emitter: ") + emitter_type);

        Emitter *emitter = new AreaLight(load_rgb(find_child_by_name(emitter_node, {"radiance"})), mesh);
        scene.m_emitters.push_back(emitter);
        mesh->m_emitter = emitter;
    }

    mesh->m_to_world_raw = Matrix4fD(load_transform(node.child("transform")));

    // add the layer instances to the scene
    for (int i = 0 ; i < layer_count ; i++){
        layer_instances[i]->m_to_world_raw = Matrix4fD(load_transform(node.child("transform")));
    }

    scene.m_meshes.push_back(mesh);

    // add the layer instances to the scene
    for (int i = 0 ; i < layer_count ; i++){
        scene.m_meshes.push_back(layer_instances[i]);
    }
}

} // namespace psdr
