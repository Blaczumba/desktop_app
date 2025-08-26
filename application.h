#pragma once

#include "bejzak_engine/common/camera/camera.h"
#include "bejzak_engine/common/camera/perspective_projection.h"
#include "bejzak_engine/common/entity_component_system/system/movement_system.h"
#include "bejzak_engine/common/input_manager/mouse_keyboard_manager.h"
#include "bejzak_engine/common/object/object.h"
#include "bejzak_engine/common/scene/octree.h"
#include "bejzak_engine/common/status/status.h"
#include "bejzak_engine/common/window/window_glfw.h"
#include "bejzak_engine/vulkan_wrapper/command_buffer/command_buffer.h"
#include "bejzak_engine/vulkan_wrapper/debug_messenger/debug_messenger.h"
#include "bejzak_engine/vulkan_wrapper/descriptor_set/bindless_descriptor_set_writer.h"
#include "bejzak_engine/vulkan_wrapper/descriptor_set/descriptor_pool.h"
#include "bejzak_engine/vulkan_wrapper/descriptor_set/descriptor_set.h"
#include "bejzak_engine/vulkan_wrapper/descriptor_set/descriptor_set_layout.h"
#include "bejzak_engine/vulkan_wrapper/descriptor_set/descriptor_set_writer.h"
#include "bejzak_engine/vulkan_wrapper/framebuffer/framebuffer.h"
#include "bejzak_engine/vulkan_wrapper/instance/instance.h"
#include "bejzak_engine/vulkan_wrapper/logical_device/logical_device.h"
#include "bejzak_engine/vulkan_wrapper/memory_objects/buffer.h"
#include "bejzak_engine/vulkan_wrapper/memory_objects/texture.h"
#include "bejzak_engine/vulkan_wrapper/model_loader/obj_loader/obj_loader.h"
#include "bejzak_engine/vulkan_wrapper/physical_device/physical_device.h"
#include "bejzak_engine/vulkan_wrapper/pipeline/graphics_pipeline.h"
#include "bejzak_engine/vulkan_wrapper/pipeline/shader_program.h"
#include "bejzak_engine/vulkan_wrapper/render_pass/render_pass.h"
#include "bejzak_engine/vulkan_wrapper/resource_manager/asset_manager.h"
#include "bejzak_engine/vulkan_wrapper/surface/surface.h"
#include "bejzak_engine/vulkan_wrapper/swapchain/swapchain.h"

#include <unordered_map>

class Application {
  static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;
  static constexpr uint32_t MAX_THREADS_IN_POOL = 2;

  Instance _instance;
#ifdef VALIDATION_LAYERS_ENABLED
  DebugMessenger _debugMessenger;
#endif // VALIDATION_LAYERS_ENABLED
  std::shared_ptr<Window> _window;
  std::unique_ptr<const MouseKeyboardManager> _mouseKeyboardManager;
  Surface _surface;
  std::unique_ptr<PhysicalDevice> _physicalDevice;
  LogicalDevice _logicalDevice;
  Swapchain _swapchain;
  std::unique_ptr<CommandPool> _singleTimeCommandPool;
  ShaderProgramManager _programManager;

  uint32_t index = 0;
  std::unordered_map<std::string, std::pair<TextureHandle, Texture>> _textures;
  std::unordered_map<std::string, Buffer> _vertexBufferMap;
  std::unordered_map<std::string, Buffer> _indexBufferMap;
  std::unordered_map<Entity, uint32_t> _entityToIndex;
  std::vector<Object> _objects;
  std::unique_ptr<Octree> _octree;
  Registry _registry;
  std::unique_ptr<AssetManager> _assetManager; // Does not have to be unique_ptr
  Renderpass _renderPass;
  std::vector<Framebuffer> _framebuffers;
  std::vector<Texture> _attachments;

  // Shadowmap
  Renderpass _shadowRenderPass;
  Framebuffer _shadowFramebuffer;
  Texture _shadowMap;
  std::unique_ptr<GraphicsPipeline>
      _shadowPipeline; // Does not have to be unique_ptr
  TextureHandle _shadowHandle;

  // Cubemap.
  Buffer _vertexBufferCube;
  Buffer _indexBufferCube;
  Texture _textureCubemap;
  VkIndexType _indexBufferCubeType;
  std::unique_ptr<GraphicsPipeline>
      _graphicsPipelineSkybox; // Does not have to be unique_ptr
  std::shared_ptr<DescriptorPool> _descriptorPoolSkybox;
  ShaderProgram _skyboxShaderProgram;
  TextureHandle _skyboxHandle;

  // PBR objects.
  std::vector<Object> objects;
  std::shared_ptr<DescriptorPool> _descriptorPool;
  std::shared_ptr<DescriptorPool> _dynamicDescriptorPool;
  ShaderProgram _pbrShaderProgram;
  std::unique_ptr<GraphicsPipeline>
      _graphicsPipeline; // Does not have to be unique_ptr
  UniformBufferCamera _ubCamera;
  UniformBufferLight _ubLight;

  DescriptorSetWriter _dynamicDescriptorSetWriter;
  Buffer _dynamicUniformBuffersCamera;
  DescriptorSet _dynamicDescriptorSet;

  ShaderProgram _shadowShaderProgram;
  std::unique_ptr<BindlessDescriptorSetWriter>
      _bindlessWriter; // Does not have to be unique_ptr

  DescriptorSet _bindlessDescriptorSet;
  Buffer _lightBuffer;
  BufferHandle _lightHandle;

  std::shared_ptr<PerspectiveProjection> _projection;
  Camera _camera;

  std::array<std::shared_ptr<CommandPool>, MAX_THREADS_IN_POOL + 1>
      _commandPools;
  std::vector<PrimaryCommandBuffer> _primaryCommandBuffer;
  std::array<std::array<SecondaryCommandBuffer, MAX_FRAMES_IN_FLIGHT>,
             MAX_THREADS_IN_POOL>
      _commandBuffers;

  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> _imageAvailableSemaphores;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> _renderFinishedSemaphores;
  std::array<VkFence, MAX_FRAMES_IN_FLIGHT> _inFlightFences;

  uint32_t _currentFrame = 0;

public:
  Application();
  ~Application();

  Application(const Application &) = delete;
  Application(Application &&) = delete;
  void operator=(const Application &) = delete;

  void run();

private:
  Status init();
  void setInput();
  void draw();
  Status createCommandBuffers();
  Status createSyncObjects();
  void updateUniformBuffer(uint32_t currentImage);
  void recordCommandBuffer(uint32_t imageIndex);
  void recordOctreeSecondaryCommandBuffer(
      const VkCommandBuffer commandBuffer, const OctreeNode *node,
      const std::array<glm::vec4, NUM_CUBE_FACES> &planes);
  void recordShadowCommandBuffer(VkCommandBuffer commandBuffer,
                                 uint32_t imageIndex);
  Status recreateSwapChain();

  Status createDescriptorSets();
  Status createPresentResources();
  Status createShadowResources();

  Status loadObjects();
  Status loadCubemap();
};
