#pragma once

#include "bejzak_engine/sources/camera/fps_camera.h"
#include "bejzak_engine/sources/command_buffer/command_buffer.h"
#include "bejzak_engine/sources/debug_messenger/debug_messenger.h"
#include "bejzak_engine/sources/descriptor_set/bindless_descriptor_set_writer.h"
#include "bejzak_engine/sources/descriptor_set/descriptor_pool.h"
#include "bejzak_engine/sources/descriptor_set/descriptor_set.h"
#include "bejzak_engine/sources/descriptor_set/descriptor_set_layout.h"
#include "bejzak_engine/sources/descriptor_set/descriptor_set_writer.h"
#include "bejzak_engine/sources/entity_component_system/system/movement_system.h"
#include "bejzak_engine/sources/framebuffer/framebuffer.h"
#include "bejzak_engine/sources/instance/instance.h"
#include "bejzak_engine/sources/logical_device/logical_device.h"
#include "bejzak_engine/sources/memory_objects/buffer.h"
#include "bejzak_engine/sources/memory_objects/texture.h"
#include "bejzak_engine/sources/model_loader/obj_loader/obj_loader.h"
#include "bejzak_engine/sources/object/object.h"
#include "bejzak_engine/sources/pipeline/graphics_pipeline.h"
#include "bejzak_engine/sources/pipeline/shader_program.h"
#include "bejzak_engine/sources/physical_device/physical_device.h"
#include "bejzak_engine/sources/render_pass/render_pass.h"
#include "bejzak_engine/sources/resource_manager/asset_manager.h"
#include "bejzak_engine/sources/scene/octree.h"
#include "bejzak_engine/sources/screenshot/screenshot.h"
#include "bejzak_engine/sources/status/status.h"
#include "bejzak_engine/sources/swapchain/swapchain.h"
#include "bejzak_engine/sources/thread_pool/thread_pool.h"
#include "bejzak_engine/sources/window/window.h"

#include <unordered_map>

class Application {
    std::unique_ptr<Instance> _instance;
#ifdef VALIDATION_LAYERS_ENABLED
    std::unique_ptr<DebugMessenger> _debugMessenger;
#endif // VALIDATION_LAYERS_ENABLED
    std::unique_ptr<Window> _window;
    std::unique_ptr<Surface> _surface;
    std::unique_ptr<PhysicalDevice> _physicalDevice;
    std::unique_ptr<LogicalDevice> _logicalDevice;
    std::unique_ptr<Swapchain> _swapchain;
    std::unique_ptr<CommandPool> _singleTimeCommandPool;
    std::unique_ptr<ShaderProgramManager> _programManager;

    uint32_t index = 0;
    std::unordered_map<std::string, std::pair<TextureHandle, Texture>> _textures;
    std::unordered_map<std::string, Buffer> _vertexBufferMap;
    std::unordered_map<std::string, Buffer> _indexBufferMap;
    std::unordered_map<Entity, uint32_t> _entityToIndex;
    std::vector<Object> _objects;
    std::unique_ptr<Octree> _octree;
    Registry _registry;
    std::unique_ptr<AssetManager> _assetManager;
    std::shared_ptr<Renderpass> _renderPass;
    std::vector<std::unique_ptr<Framebuffer>> _framebuffers;
    std::vector<Texture> _attachments;

    // Shadowmap
    std::shared_ptr<Renderpass> _shadowRenderPass;
    std::unique_ptr<Framebuffer> _shadowFramebuffer;
    Texture _shadowMap;
    std::unique_ptr<GraphicsPipeline> _shadowPipeline;
    TextureHandle _shadowHandle;

    // Cubemap.
    Buffer _vertexBufferCube;
    Buffer _indexBufferCube;
    Texture _textureCubemap;
    VkIndexType _indexBufferCubeType;
    std::unique_ptr<GraphicsPipeline> _graphicsPipelineSkybox;
    std::shared_ptr<DescriptorPool> _descriptorPoolSkybox;
    std::unique_ptr<ShaderProgram> _skyboxShaderProgram;
    TextureHandle _skyboxHandle;

    // PBR objects.
    std::vector<Object> objects;
    std::shared_ptr<DescriptorPool> _descriptorPool;
    std::shared_ptr<DescriptorPool> _dynamicDescriptorPool;
    std::unique_ptr<ShaderProgram> _pbrShaderProgram;
    std::unique_ptr<GraphicsPipeline> _graphicsPipeline;
    UniformBufferCamera _ubCamera;
    UniformBufferLight _ubLight;

    DescriptorSetWriter _dynamicDescriptorSetWriter;
    Buffer _dynamicUniformBuffersCamera;
    DescriptorSet _dynamicDescriptorSet;

    std::unique_ptr<ShaderProgram> _shadowShaderProgram;
    std::unique_ptr<BindlessDescriptorSetWriter> _bindlessWriter;

    DescriptorSet _bindlessDescriptorSet;
    Buffer _lightBuffer;
    BufferHandle _lightHandle;

    std::unique_ptr<FPSCamera> _camera;

    std::vector<std::shared_ptr<CommandPool>> _commandPool;
    std::vector<std::unique_ptr<PrimaryCommandBuffer>> _primaryCommandBuffer;
    std::vector<std::vector<std::unique_ptr<SecondaryCommandBuffer>>> _commandBuffers;
    std::vector<std::vector<std::unique_ptr<SecondaryCommandBuffer>>> _shadowCommandBuffers;

    std::vector<VkSemaphore> _shadowMapSemaphores;
    std::vector<VkSemaphore> _imageAvailableSemaphores;
    std::vector<VkSemaphore> _renderFinishedSemaphores;
    std::vector<VkFence> _inFlightFences;

    uint32_t _currentFrame = 0;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;
    static constexpr uint32_t MAX_THREADS_IN_POOL = 2;

    float _mouseXOffset;
    float _mouseYOffset;

public:
    Application();
    ~Application();

    Application(const Application&) = delete;
    Application(Application&&) = delete;
    void operator=(const Application&) = delete;

    void run();
private:
	Status init();
    void setInput();
    void draw();
    Status createCommandBuffers();
    void createSyncObjects();
    void updateUniformBuffer(uint32_t currentImage);
    void recordCommandBuffer(uint32_t imageIndex);
    void recordOctreeSecondaryCommandBuffer(const VkCommandBuffer commandBuffer, const OctreeNode* node, const std::array<glm::vec4, NUM_CUBE_FACES>& planes);
    void recordShadowCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    Status recreateSwapChain();

    Status createDescriptorSets();
    Status createPresentResources();
    Status createShadowResources();

    Status loadObjects();
    Status loadCubemap();
};
