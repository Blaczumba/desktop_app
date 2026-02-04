#pragma once

#include "bejzak_engine/common/camera/camera.h"
#include "bejzak_engine/common/camera/projection.h"
#include "bejzak_engine/common/entity_component_system/system/movement_system.h"
#include "bejzak_engine/common/input_manager/mouse_keyboard_manager.h"
#include "bejzak_engine/common/model_loader/model_loader.h"
#include "bejzak_engine/common/object/object.h"
#include "bejzak_engine/common/scene/octree.h"
#include "bejzak_engine/common/util/primitives.h"
#include "bejzak_engine/common/window/window_glfw.h"
#include "bejzak_engine/vulkan/resource_manager/asset_manager.h"
#include "bejzak_engine/vulkan/resource_manager/gpu_buffer_manager.h"
#include "bejzak_engine/vulkan/resource_manager/pipeline_manager.h"
#include "bejzak_engine/vulkan/resource_manager/sampler_manager.h"
#include "bejzak_engine/vulkan/wrapper/command_buffer/command_buffer.h"
#include "bejzak_engine/vulkan/wrapper/debug_messenger/debug_messenger.h"
#include "bejzak_engine/vulkan/resource_manager/bindless_descriptor_set_writer.h"
#include "bejzak_engine/vulkan/wrapper/descriptor_set/descriptor_pool.h"
#include "bejzak_engine/vulkan/wrapper/descriptor_set/descriptor_set.h"
#include "bejzak_engine/vulkan/wrapper/descriptor_set/descriptor_set_layout.h"
#include "bejzak_engine/vulkan/wrapper/descriptor_set/descriptor_set_writer.h"
#include "bejzak_engine/vulkan/wrapper/framebuffer/framebuffer.h"
#include "bejzak_engine/vulkan/wrapper/instance/instance.h"
#include "bejzak_engine/vulkan/wrapper/logical_device/logical_device.h"
#include "bejzak_engine/vulkan/wrapper/memory_objects/buffer.h"
#include "bejzak_engine/vulkan/wrapper/memory_objects/texture.h"
#include "bejzak_engine/vulkan/wrapper/physical_device/physical_device.h"
#include "bejzak_engine/vulkan/wrapper/render_pass/render_pass.h"
#include "bejzak_engine/vulkan/wrapper/surface/surface.h"
#include "bejzak_engine/vulkan/wrapper/swapchain/swapchain.h"

#include <unordered_map>

class Application {
  static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;
  static constexpr uint32_t MAX_THREADS_IN_POOL = 2;

  Instance _instance;
  DebugMessenger _debugMessenger;
  std::shared_ptr<Window> _window;
  std::unique_ptr<const MouseKeyboardManager> _mouseKeyboardManager;
  Surface _surface;
  std::unique_ptr<PhysicalDevice> _physicalDevice;
  LogicalDevice _logicalDevice;
  Swapchain _swapchain;
  std::unique_ptr<CommandPool> _singleTimeCommandPool;

  std::unique_ptr<PipelineManager> _pipelineManager;

  std::vector<Object> _objects;
  std::unique_ptr<Octree> _octree;
  Registry _registry;
  std::unique_ptr<AssetManager> _assetManager;
  std::unique_ptr<GpuBufferManager> _gpuBufferManager;
  std::unique_ptr<SamplerManager> _samplerManager;

  Renderpass _renderPass;
  std::vector<Framebuffer> _framebuffers;
  std::vector<Texture> _attachments;

  // Shadowmap
  Renderpass _shadowRenderPass;
  Framebuffer _shadowFramebuffer;
  Texture _shadowMap;
  Pipeline *_shadowPipeline;
  UniformTextureHandle _shadowHandle;

  // Skybox.
  GpuBufferHandle _vertexBufferCubeHandle;
  GpuBufferHandle _vertexBufferCubeNormalsHandle;
  GpuBufferHandle _indexBufferCubeHandle;
  Texture _textureCubemap;
  VkIndexType _indexBufferCubeType;
  Pipeline *_skyboxPipeline;
  UniformTextureHandle _skyboxHandle;

  // Mirror cubemap
  // First pass.
  Renderpass _envMappingRenderPass;
  Framebuffer _envMappingFramebuffer;
  Pipeline *_envMappingPipeline;
  Buffer _envMappingUniformBuffer;
  UniformBufferHandle _envMappingHandle;
  std::array<Texture, 2> _envMappingAttachments;
  UniformTextureHandle _envMappingTextureHandle;
  // Second pass.
  Pipeline *_phongEnvMappingPipeline;

  // PBR objects.
  std::vector<Object> objects;
  std::shared_ptr<DescriptorPool> _descriptorPool;
  std::shared_ptr<DescriptorPool> _dynamicDescriptorPool;
  Pipeline *_graphicsPipeline;
  UniformBufferCamera _ubCamera;
  UniformBufferLight _ubLight;

  DescriptorSetWriter _dynamicDescriptorSetWriter;
  Buffer _dynamicUniformBuffersCamera;
  DescriptorSet _dynamicDescriptorSet;

  std::unique_ptr<BindlessDescriptorSetWriter> _bindlessWriter;

  DescriptorSet _bindlessDescriptorSet;
  Buffer _lightBuffer;
  UniformBufferHandle _lightHandle;

  Camera _camera;

  std::array<std::shared_ptr<CommandPool>, MAX_THREADS_IN_POOL + 1>
      _commandPools;
  std::array<CommandBuffer, MAX_FRAMES_IN_FLIGHT> _primaryCommandBuffer;
  std::array<std::array<CommandBuffer, MAX_FRAMES_IN_FLIGHT>,
             MAX_THREADS_IN_POOL>
      _secondaryCommandBuffers;

  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> _imageAvailableSemaphores;
  std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> _renderFinishedSemaphores;
  std::array<VkFence, MAX_FRAMES_IN_FLIGHT> _inFlightFences;

  uint32_t _currentFrame = 0;

  std::unique_ptr<FileLoader> _fileLoader;

public:
  Application(std::unique_ptr<FileLoader> &&fileLoader);
  ~Application();

  Application(const Application &) = delete;
  Application(Application &&) = delete;
  void operator=(const Application &) = delete;

  void run();

private:
  void init();
  void setInput();
  void draw();
  void createCommandBuffers();
  void createSyncObjects();
  void updateUniformBuffer(uint32_t currentImage);
  void recordCommandBuffer(uint32_t imageIndex);
  void recordOctreeSecondaryCommandBuffer(const VkCommandBuffer commandBuffer,
                                          const OctreeNode *node,
                                          std::span<const glm::vec4> planes);
  void recordShadowCommandBuffer(VkCommandBuffer commandBuffer);
  void recordEnvMappingCommandBuffer(VkCommandBuffer commandBuffer);
  void recreateSwapChain();

  void createDescriptorSets();
  void createGraphicsPipelines();
  void createPresentResources();
  void createShadowResources();
  void createEnvMappingResources();

  void loadObjects(std::span<const VertexData> sceneData);
  std::tuple<UniformTextureHandle, GpuTextureHandle>
  getOrLoadTexture(
      std::unordered_map<StagingImageDataResourceHandle,
                         std::pair<UniformTextureHandle,
                                   GpuTextureHandle>>
          &textureCache,
      StagingImageDataResourceHandle textureID, VkFormat format,
      VkCommandBuffer commandBuffer, float maxSamplerAnisotropy, SamplerHandle samplerHandle);
  void createOctreeScene();
  void loadCubemap(const VertexData &cubeData);
};
