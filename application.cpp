#include "application.h"

#include "bejzak_engine/common/camera/camera.h"
#include "bejzak_engine/common/camera/projection.h"
#include "bejzak_engine/common/entity_component_system/component/material.h"
#include "bejzak_engine/common/entity_component_system/component/mesh.h"
#include "bejzak_engine/common/entity_component_system/component/position.h"
#include "bejzak_engine/common/entity_component_system/component/transform.h"
#include "bejzak_engine/common/entity_component_system/component/velocity.h"
#include "bejzak_engine/common/entity_component_system/system/movement_system.h"
#include "bejzak_engine/common/file/standard_file_loader.h"
#include "bejzak_engine/common/model_loader/model_loader.h"
#include "bejzak_engine/common/model_loader/obj_loader/obj_loader.h"
#include "bejzak_engine/common/model_loader/tiny_gltf_loader/tiny_gltf_loader.h"
#include "bejzak_engine/common/window/window_glfw.h"
#include "bejzak_engine/lib/buffer/shared_buffer.h"
#include "bejzak_engine/vulkan/resource_manager/pipeline_manager.h"
#include "bejzak_engine/vulkan/wrapper/pipeline/input_description.h"
#include "bejzak_engine/vulkan/wrapper/render_pass/attachment_layout.h"
#include "bejzak_engine/vulkan/wrapper/util/check.h"
#include "bejzak_engine/lib/inplace_vector/inplace_vector.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <print>
#include <queue>

namespace {

lib::Buffer<VkBufferImageCopy>
createBufferImageCopyRegions(std::span<const ImageSubresource> subresources) {
  lib::Buffer<VkBufferImageCopy> regions(subresources.size());
  std::transform(
      subresources.cbegin(), subresources.cend(), regions.begin(),
      [](const ImageSubresource &subresource) {
        return VkBufferImageCopy{
            .bufferOffset = subresource.offset,
            .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .mipLevel = subresource.mipLevel,
                                 .baseArrayLayer = subresource.baseArrayLayer,
                                 .layerCount = subresource.layerCount},
            .imageExtent = {.width = subresource.width,
                            .height = subresource.height,
                            .depth = subresource.depth}};
      });
  return regions;
}

Texture createSkybox(const LogicalDevice &logicalDevice,
                     VkCommandBuffer commandBuffer,
                     const AssetManager::ImageData &imageData, VkFormat format,
                     float samplerAnisotropy) {

  Texture texture =
      TextureBuilder()
          .withAspect(VK_IMAGE_ASPECT_COLOR_BIT)
          .withExtent(imageData.width, imageData.height)
          .withFormat(format)
          .withMipLevels(imageData.mipLevels)
          .withUsage(VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT)
          .withLayerCount(6)
          .withAdditionalCreateInfoFlags(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
          .withMaxAnisotropy(samplerAnisotropy)
          .withMaxLod(static_cast<float>(imageData.mipLevels))
          .withLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
          .buildImage(logicalDevice, commandBuffer,
                      imageData.stagingBuffer.getVkBuffer(),
                      createBufferImageCopyRegions(imageData.copyRegions));
  texture.addCreateVkImageView(0, imageData.mipLevels, 0, 6);
  return texture;
}

Texture createCubemap(const LogicalDevice &logicalDevice,
                      VkCommandBuffer commandBuffer,

                      VkImageAspectFlags aspect, VkFormat format,
                      VkImageUsageFlags additionalUsage,
                      float samplerAnisotropy) {
  Texture texture =
      TextureBuilder()
          .withAspect(aspect)
          .withExtent(1024 * 4, 1024 * 4)
          .withFormat(format)
          .withUsage(VK_IMAGE_USAGE_SAMPLED_BIT | additionalUsage)
          .withLayerCount(6)
          .withAdditionalCreateInfoFlags(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
          .withMaxAnisotropy(samplerAnisotropy)
          .withNumSamples(VK_SAMPLE_COUNT_1_BIT)
          .withMipmapMode(VK_SAMPLER_MIPMAP_MODE_NEAREST)
          .buildAttachment(logicalDevice, commandBuffer);
  texture.addCreateVkImageView(0, 1, 0, 6);
  return texture;
}

Texture createShadowmap(const LogicalDevice &logicalDevice,
                        VkCommandBuffer commandBuffer, uint32_t width,
                        uint32_t height, VkFormat format) {

  Texture texture =
      TextureBuilder()
          .withAspect(VK_IMAGE_ASPECT_DEPTH_BIT)
          .withExtent(width, height)
          .withFormat(format)
          .withUsage(VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
          .withAddressModes(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)
          .withCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL)
          .withBorderColor(VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE)
          .buildImageSampler(logicalDevice, commandBuffer);
  texture.addCreateVkImageView(0, 1, 0, 1);
  return texture;
}

Texture createTexture2D(const LogicalDevice &logicalDevice,
                        VkCommandBuffer commandBuffer,
                        const AssetManager::ImageData &imageData,
                        VkFormat format, float samplerAnisotropy) {
  Texture texture = TextureBuilder()
                        .withAspect(VK_IMAGE_ASPECT_COLOR_BIT)
                        .withExtent(imageData.width, imageData.height)
                        .withFormat(format)
                        .withMipLevels(imageData.mipLevels)
                        .withUsage(VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT)
                        .withMaxAnisotropy(samplerAnisotropy)
                        .withMaxLod(static_cast<float>(imageData.mipLevels))
                        .buildMipmapImage(logicalDevice, commandBuffer,
                                          imageData.stagingBuffer.getVkBuffer(),
                                          createBufferImageCopyRegions(
                                              imageData.copyRegions));
  texture.addCreateVkImageView(0, imageData.mipLevels, 0, 1);
  return texture;
}

} // namespace

Application::Application(std::unique_ptr<FileLoader>&& fileLoader)
    : _camera(PerspectiveProjection{glm::radians(45.0f), 1920.0f / 1080.f,
                                    0.01f, 50.0f},
              glm::vec3(0.0f), 5.5f, 0.01f),
      _pipelineManager(PipelineManager::create(*fileLoader)), _fileLoader(std::move(fileLoader)) {
  init();
  _assetManager = AssetManager::create(_logicalDevice, *_fileLoader);
  // Load data from disk.
  std::string data = _fileLoader->loadFileToString(MODELS_PATH "cube.obj");
  VertexData cubeData = loadObj(*_assetManager, "cube.obj", data);
  const std::vector<VertexData> sceneData =
      LoadGltfFromFile(*_assetManager, MODELS_PATH "sponza/scene.gltf");
  cubeData.diffuseTexture = { _assetManager->loadImageAsync(TEXTURES_PATH "cubemap_yokohama_rgba.ktx"), TEXTURES_PATH "cubemap_yokohama_rgba.ktx"};
  loadCubemap(cubeData);
  createDescriptorSets();
  createPresentResources();
  createEnvMappingResources();
  createShadowResources();
  createGraphicsPipelines();
  createCommandBuffers();
  createSyncObjects();
  loadObjects(sceneData);
  createOctreeScene();
  setInput();
}

VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  std::cerr << "[Vulkan Validation] " << "Severity: " << messageSeverity << ", "
            << "Type: " << messageType << std::endl
            << "Message: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

void Application::init() {
  _window = std::make_unique<WindowGlfw>("Bejzak Engine", 1920, 1080);
  _mouseKeyboardManager = _window->createMouseKeyboardManager();
  std::vector<const char *> requiredExtensions = _window->getVulkanExtensions();
#ifdef VALIDATION_LAYERS_ENABLED
  requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif // VALIDATION_LAYERS_ENABLED
  requiredExtensions.push_back(
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  _instance =
      Instance::create("Bejzak Engine", requiredExtensions, debugCallback);
#ifdef VALIDATION_LAYERS_ENABLED
  _debugMessenger = DebugMessenger::create(_instance, debugCallback);
#endif // VALIDATION_LAYERS_ENABLED

  _surface = Surface::create(_instance, *_window);
  _physicalDevice = PhysicalDevice::create(_instance, _surface.getVkSurface());
  _logicalDevice = LogicalDevice::create(*_physicalDevice);
  const Extent2D framebufferSize = _window->getFramebufferSize();
  _swapchain =
      SwapchainBuilder()
          .withPreferredPresentMode(VK_PRESENT_MODE_MAILBOX_KHR)
          .build(_logicalDevice, _surface.getVkSurface(),
                 VkExtent2D{framebufferSize.width, framebufferSize.height});
  _singleTimeCommandPool =
      CommandPool::create(_logicalDevice, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
}

void Application::setInput() {
  if (_mouseKeyboardManager == nullptr) {
    return;
  }

  _mouseKeyboardManager->absorbCursor();
  _mouseKeyboardManager->setKeyboardCallback(
      [&](Keyboard::Key key, int action) {
        switch (key) {
        case Keyboard::Key::Escape:
          _window->close();
          break;
        }
      });
}

void Application::createEnvMappingResources() {
  // First pass for rendering the environment map.
  const float samplerAnisotropy = _physicalDevice->getMaxSamplerAnisotropy();
  {
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);

    _envMappingAttachments[0] =
        createCubemap(_logicalDevice, handle.getCommandBuffer(),
                      VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R8G8B8A8_SRGB,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, samplerAnisotropy);
    _envMappingAttachments[1] = createCubemap(
        _logicalDevice, handle.getCommandBuffer(), VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_FORMAT_D16_UNORM, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        samplerAnisotropy);
  }

  AttachmentLayout attachmentLayout;
  attachmentLayout.addColorAttachment(VK_FORMAT_R8G8B8A8_SRGB,
                                      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                      VK_ATTACHMENT_STORE_OP_STORE);
  attachmentLayout.addDepthAttachment(VK_FORMAT_D16_UNORM,
                                      VK_ATTACHMENT_STORE_OP_DONT_CARE);

  _envMappingRenderPass =
      RenderpassBuilder(attachmentLayout)
          .withMultiView({0b111111}, {0b111111})
          .addDependency(VK_SUBPASS_EXTERNAL, 0,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
          .addSubpass({0, 1})
          .build(_logicalDevice);

  _envMappingFramebuffer = Framebuffer::createFromTextures(
      _envMappingRenderPass, _envMappingAttachments);

  const glm::vec3 pos = glm::vec3(0.0f, 2.0f, 0.0f);
  glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 50.0f);

  struct {
    alignas(16) glm::mat4 projView[6];
    alignas(16) glm::vec3 viewPos;
    alignas(16) glm::mat4 lightProjView;
    alignas(16) glm::vec3 lightPos;
  } const faceTransform = {
      .projView =
          {
              proj * glm::lookAt(pos, pos + glm::vec3(1.0f, 0.0f, 0.0f),
                                 glm::vec3(0.0f, -1.0f, 0.0f)),
              proj * glm::lookAt(pos, pos + glm::vec3(-1.0f, 0.0f, 0.0f),
                                 glm::vec3(0.0f, -1.0f, 0.0f)),
              proj * glm::lookAt(pos, pos + glm::vec3(0.0f, 1.0f, 0.0f),
                                 glm::vec3(0.0f, 0.0f, 1.0f)),
              proj * glm::lookAt(pos, pos + glm::vec3(0.0f, -1.0f, 0.0f),
                                 glm::vec3(0.0f, 0.0f, -1.0f)),
              proj * glm::lookAt(pos, pos + glm::vec3(0.0f, 0.0f, 1.0f),
                                 glm::vec3(0.0f, -1.0f, 0.0f)),
              proj * glm::lookAt(pos, pos + glm::vec3(0.0f, 0.0f, -1.0f),
                                 glm::vec3(0.0f, -1.0f, 0.0f)),
          },
      .viewPos = pos,
      .lightProjView = _ubLight.projView,
      .lightPos = _ubLight.pos};

  _envMappingUniformBuffer =
      Buffer::createUniformBuffer(_logicalDevice, sizeof(faceTransform));
  _envMappingUniformBuffer.copyData(faceTransform);
  _envMappingHandle =
      _bindlessWriter->storeBuffer(_envMappingUniformBuffer);
  _envMappingTextureHandle =
      _bindlessWriter->storeTexture(_envMappingAttachments[0]);
}

void Application::loadCubemap(const VertexData& cubeData) {
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    const VkCommandBuffer commandBuffer = handle.getCommandBuffer();

    const AssetManager::ImageData &imageData =
        _assetManager->getImageData(cubeData.diffuseTexture.ID);

    _textureCubemap = createSkybox(_logicalDevice, commandBuffer, imageData,
                                   VK_FORMAT_R8G8B8A8_UNORM,
                                   _physicalDevice->getMaxSamplerAnisotropy());

    const AssetManager::VertexData &vData =
        _assetManager->getVertexData(cubeData.vertexResourceID);
    _vertexBufferCube = Buffer::createVertexBuffer(
        _logicalDevice, vData.buffers.at("P").getSize());
    _vertexBufferCube.copyBuffer(commandBuffer, vData.buffers.at("P"));

    _vertexBufferCubeNormals = Buffer::createVertexBuffer(
        _logicalDevice, vData.buffers.at("PN").getSize());
    _vertexBufferCubeNormals.copyBuffer(commandBuffer, vData.buffers.at("PN"));

    _indexBufferCube =
        Buffer::createIndexBuffer(_logicalDevice, vData.indexBuffer.getSize());

    _indexBufferCube.copyBuffer(commandBuffer, vData.indexBuffer);
    _indexBufferCubeType = vData.indexType;
}

void Application::loadObjects(std::span<const VertexData> sceneData) {
  const float maxSamplerAnisotropy = _physicalDevice->getMaxSamplerAnisotropy();
  _objects.reserve(sceneData.size());

  SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
  const VkCommandBuffer commandBuffer = handle.getCommandBuffer();
  for (const VertexData &sceneObject : sceneData) {
    const std::string diffusePath =
        MODELS_PATH "sponza/" + sceneObject.diffuseTexture.path;
    if (!_textures.contains(diffusePath)) {
      const AssetManager::ImageData &imgData =
          _assetManager->getImageData(sceneObject.diffuseTexture.ID);
      Texture texture =
          createTexture2D(_logicalDevice, commandBuffer, imgData,
                          VK_FORMAT_R8G8B8A8_SRGB, maxSamplerAnisotropy);
      _textures.emplace(diffusePath,
                        std::make_pair(_bindlessWriter->storeTexture(texture),
                                       std::move(texture)));
    }

    const std::string normalPath =
        MODELS_PATH "sponza/" + sceneObject.normalTexture.path;
    if (!_textures.contains(normalPath)) {
      const AssetManager::ImageData &imgData =
          _assetManager->getImageData(sceneObject.normalTexture.ID);
      Texture texture =
          createTexture2D(_logicalDevice, commandBuffer, imgData,
                          VK_FORMAT_R8G8B8A8_UNORM, maxSamplerAnisotropy);
      _textures.emplace(normalPath,
                        std::make_pair(_bindlessWriter->storeTexture(texture),
                                       std::move(texture)));
    }

    const std::string metallicRoughnessPath =
        MODELS_PATH "sponza/" + sceneObject.metallicRoughnessTexture.path;
    if (!_textures.contains(metallicRoughnessPath)) {
      const AssetManager::ImageData &imgData =
          _assetManager->getImageData(sceneObject.metallicRoughnessTexture.ID);
      Texture texture =
          createTexture2D(_logicalDevice, commandBuffer, imgData,
                          VK_FORMAT_R8G8B8A8_UNORM, maxSamplerAnisotropy);
      _textures.emplace(metallicRoughnessPath,
                        std::make_pair(_bindlessWriter->storeTexture(texture),
                                       std::move(texture)));
    }

    Entity e = _registry.createEntity();
    _objects.emplace_back("", e);
    _registry.addComponent<MaterialComponent>(
        e, MaterialComponent{_textures[diffusePath].first,
                             _textures[normalPath].first,
                             _textures[metallicRoughnessPath].first});
    const AssetManager::VertexData &vData =
        _assetManager->getVertexData(sceneObject.vertexResourceID);
    MeshComponent msh;
    msh.vertexBuffer = Buffer::createVertexBuffer(
        _logicalDevice, vData.buffers.at("PTNT").getSize());

    msh.vertexBuffer.copyBuffer(commandBuffer, vData.buffers.at("PTNT"));

    msh.indexBuffer =
        Buffer::createIndexBuffer(_logicalDevice, vData.indexBuffer.getSize());

    msh.indexBuffer.copyBuffer(commandBuffer, vData.indexBuffer);
    msh.vertexBufferPrimitive = Buffer::createVertexBuffer(
        _logicalDevice, vData.buffers.at("P").getSize());
    msh.vertexBufferPrimitive.copyBuffer(commandBuffer, vData.buffers.at("P"));
    msh.indexType = vData.indexType;
    msh.aabb = createAABBfromVertices(sceneObject.positions, sceneObject.model);
    _registry.addComponent<MeshComponent>(e, std::move(msh));

    TransformComponent trsf;
    trsf.model = sceneObject.model;
    _registry.addComponent<TransformComponent>(e, std::move(trsf));
  }
}

void Application::createOctreeScene() {
  AABB sceneAABB =
      _registry.getComponent<MeshComponent>(_objects[0].getEntity()).aabb;

  for (int i = 1; i < _objects.size(); ++i) {
    sceneAABB.extend(
        _registry.getComponent<MeshComponent>(_objects[i].getEntity()).aabb);
  }
  _octree = std::make_unique<Octree>(sceneAABB);

  for (const Object &object : _objects)
    _octree->addObject(
        &object,
        _registry.getComponent<MeshComponent>(object.getEntity()).aabb);
}

void Application::createDescriptorSets() {
  const uint32_t size = _logicalDevice.getPhysicalDevice().getMemoryAlignment(
      sizeof(UniformBufferCamera));

  _dynamicUniformBuffersCamera =
      Buffer::createUniformBuffer(_logicalDevice, MAX_FRAMES_IN_FLIGHT * size);

  _descriptorPool = DescriptorPool::create(
      _logicalDevice, 150, VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT);
  _dynamicDescriptorPool = DescriptorPool::create(_logicalDevice, 1);

  const VkDescriptorSetLayout bindlesslayout =
      _pipelineManager->getOrCreateBindlessLayout(_logicalDevice);
  _bindlessDescriptorSet = _descriptorPool->createDesriptorSet(bindlesslayout);

  const VkDescriptorSetLayout cameraLayout =
      _pipelineManager->getOrCreateCameraLayout(_logicalDevice);
  _dynamicDescriptorSet =
      _dynamicDescriptorPool->createDesriptorSet(cameraLayout);
  _bindlessWriter =
      std::make_unique<BindlessDescriptorSetWriter>(_bindlessDescriptorSet);
  _skyboxHandle = _bindlessWriter->storeTexture(_textureCubemap);

  _dynamicDescriptorSetWriter.storeDynamicBuffer(_dynamicUniformBuffersCamera,
                                                 size);
  _dynamicDescriptorSetWriter.writeDescriptorSet(
      _logicalDevice.getVkDevice(), _dynamicDescriptorSet.getVkDescriptorSet());

  _lightBuffer =
      Buffer::createUniformBuffer(_logicalDevice, sizeof(UniformBufferLight));
  _lightHandle = _bindlessWriter->storeBuffer(_lightBuffer);

  _ubLight.pos = glm::vec3(15.1891f, 2.66408f, -0.841221f);
  _ubLight.projView = glm::perspective(glm::radians(120.0f), 1.0f, 0.1f, 40.0f);
  _ubLight.projView[1][1] = -_ubLight.projView[1][1];
  _ubLight.projView =
      _ubLight.projView * glm::lookAt(_ubLight.pos,
                                      glm::vec3(-3.82383f, 3.66503f, 1.30751f),
                                      glm::vec3(0.0f, 1.0f, 0.0f));
  _lightBuffer.copyData(_ubLight, 0);
}

void Application::createGraphicsPipelines() {
  _graphicsPipeline = _pipelineManager->getPipeline(_pipelineManager->createPBRProgram(_renderPass));
  _skyboxPipeline = _pipelineManager->getPipeline(_pipelineManager->createSkyboxProgram(_renderPass));
  _phongEnvMappingPipeline = _pipelineManager->getPipeline(_pipelineManager->createEnvMappingProgram(_renderPass));
  _shadowPipeline = _pipelineManager->getPipeline(_pipelineManager->createShadowProgram(_shadowRenderPass));
  _envMappingPipeline = _pipelineManager->getPipeline(_pipelineManager->createPbrEnvMappingProgram(_envMappingRenderPass));
}

void Application::createPresentResources() {
  static constexpr VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_4_BIT;
  const VkFormat swapchainImageFormat = _swapchain.getVkFormat();

  AttachmentLayout attachmentsLayout(msaaSamples);
  attachmentsLayout
      .addColorResolvePresentAttachment(swapchainImageFormat,
                                        VK_ATTACHMENT_LOAD_OP_DONT_CARE)
      .addColorAttachment(swapchainImageFormat, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                          VK_ATTACHMENT_STORE_OP_DONT_CARE)
      .addDepthAttachment(VK_FORMAT_D24_UNORM_S8_UINT,
                          VK_ATTACHMENT_STORE_OP_DONT_CARE);

  _renderPass =
      RenderpassBuilder(attachmentsLayout)
          .addDependency(VK_SUBPASS_EXTERNAL, 0,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
          .addSubpass({0, 1, 2})
          .build(_logicalDevice);

  {
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    const VkCommandBuffer commandBuffer = handle.getCommandBuffer();
    for (uint8_t i = 0; i < _swapchain.getImagesCount(); ++i) {
      _framebuffers.push_back(Framebuffer::createFromSwapchain(
          commandBuffer, _renderPass, _swapchain.getExtent(),
          _swapchain.getSwapchainVkImageView(i), _attachments));
    }
  }
}

void Application::createShadowResources() {
  {
    // TODO: Should not be in this function.
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    const VkCommandBuffer commandBuffer = handle.getCommandBuffer();
    _shadowMap = createShadowmap(_logicalDevice, commandBuffer, 1024 * 2,
                                 1024 * 2, VK_FORMAT_D32_SFLOAT);
  }
  _shadowHandle = _bindlessWriter->storeTexture(_shadowMap);

  AttachmentLayout attachmentLayout;
  attachmentLayout.addShadowAttachment(
      VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  _shadowRenderPass =
      RenderpassBuilder(attachmentLayout).addSubpass({0}).build(_logicalDevice);
  _shadowFramebuffer = Framebuffer::createFromTextures(
      _shadowRenderPass, std::span(&_shadowMap, 1));
}

Application::~Application() {
  const VkDevice device = _logicalDevice.getVkDevice();

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(device, _renderFinishedSemaphores[i], nullptr);
    vkDestroySemaphore(device, _imageAvailableSemaphores[i], nullptr);
    vkDestroyFence(device, _inFlightFences[i], nullptr);
  }
}

void Application::run() {
  updateUniformBuffer(_currentFrame);
  {
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    recordShadowCommandBuffer(handle.getCommandBuffer());
    recordEnvMappingCommandBuffer(handle.getCommandBuffer());
  }
  std::chrono::steady_clock::time_point previous;

  while (_window->open()) {
    const std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    const float deltaTime =
        std::chrono::duration<float>(now - previous).count();
    std::println("{}", 1.0f / deltaTime);
    previous = now;
    _window->pollEvents();
    _camera.updateFromKeyboard(*_mouseKeyboardManager, deltaTime);
    draw();
  }
  vkDeviceWaitIdle(_logicalDevice.getVkDevice());
}

void Application::draw() {
  vkWaitForFences(_logicalDevice.getVkDevice(), 1,
                  &_inFlightFences[_currentFrame], VK_TRUE, UINT64_MAX);

  uint32_t imageIndex;
  VkResult result = _swapchain.acquireNextImage(
      _imageAvailableSemaphores[_currentFrame], &imageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    recreateSwapChain();
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  updateUniformBuffer(_currentFrame);

  vkResetFences(_logicalDevice.getVkDevice(), 1,
                &_inFlightFences[_currentFrame]);

  recordCommandBuffer(imageIndex);

  VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};

  VkSemaphore waitSemaphores[] = {_imageAvailableSemaphores[_currentFrame]};
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  VkCommandBuffer submitCommands[] = {
      _primaryCommandBuffer[_currentFrame].getVkCommandBuffer()};
  submitInfo.commandBufferCount =
      static_cast<uint32_t>(std::size(submitCommands));
  submitInfo.pCommandBuffers = submitCommands;

  VkSemaphore signalSemaphores[] = {_renderFinishedSemaphores[_currentFrame]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(_logicalDevice.getGraphicsVkQueue(), 1, &submitInfo,
                    _inFlightFences[_currentFrame]) != VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  result = _swapchain.present(imageIndex, signalSemaphores[0]);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    recreateSwapChain();
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  if (++_currentFrame == MAX_FRAMES_IN_FLIGHT) {
    _currentFrame = 0;
  }
}

void Application::createSyncObjects() {
  static constexpr VkSemaphoreCreateInfo semaphoreInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  static constexpr VkFenceCreateInfo fenceInfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT};

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    CHECK_VKCMD(vkCreateSemaphore(_logicalDevice.getVkDevice(), &semaphoreInfo,
                                  nullptr, &_imageAvailableSemaphores[i]),
                "Failed to create VkSemaphore.");
    CHECK_VKCMD(vkCreateSemaphore(_logicalDevice.getVkDevice(), &semaphoreInfo,
                                  nullptr, &_renderFinishedSemaphores[i]),
                "Failed to create VkSemaphore.");
    CHECK_VKCMD(vkCreateFence(_logicalDevice.getVkDevice(), &fenceInfo, nullptr,
                              &_inFlightFences[i]),
                "Failed to create VkFence.");
  }
}

void Application::updateUniformBuffer(uint32_t currentFrame) {
  _ubCamera.view = _camera.getViewMatrix();
  _ubCamera.proj = _camera.getProjectionMatrix();
  _ubCamera.pos = _camera.getPosition();
  _dynamicUniformBuffersCamera.copyData(
      _ubCamera, currentFrame * _physicalDevice->getMemoryAlignment(
                                    sizeof(UniformBufferCamera)));
}

void Application::createCommandBuffers() {
  for (int i = 0; i <= MAX_THREADS_IN_POOL; i++) {

    _commandPools[i] = CommandPool::create(
        _logicalDevice, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  }
  _primaryCommandBuffer = _commandPools[MAX_THREADS_IN_POOL]
                              ->createCommandBuffers<MAX_FRAMES_IN_FLIGHT>(
                                  VK_COMMAND_BUFFER_LEVEL_PRIMARY);
  for (int i = 0; i < MAX_THREADS_IN_POOL; i++) {
    _secondaryCommandBuffers[i] =
        _commandPools[i]->createCommandBuffers<MAX_FRAMES_IN_FLIGHT>(
            VK_COMMAND_BUFFER_LEVEL_SECONDARY);
  }
}

void Application::recordOctreeSecondaryCommandBuffer(
    const VkCommandBuffer commandBuffer, const OctreeNode *rootNode,
    std::span<const glm::vec4> planes) {
  if (!rootNode || !rootNode->getVolume().intersectsFrustum(planes))
    return;

  static std::queue<const OctreeNode *> nodeQueue; // Keep it static to preserve
  // capacity
  nodeQueue.push(rootNode);

  while (!nodeQueue.empty()) {
    const OctreeNode *node = nodeQueue.front();
    nodeQueue.pop();

    for (const Object *object : node->getObjects()) {

      const auto &materialComponent =
          _registry.getComponent<MaterialComponent>(object->getEntity());
      const auto &transformComponent =
          _registry.getComponent<TransformComponent>(object->getEntity());

      const PushConstantsModelDescriptorHandles pc = {
        .model = transformComponent.model,
        .descriptorHandles = {
            static_cast<uint32_t>(_lightHandle),
            static_cast<uint32_t>(materialComponent.diffuse),
            static_cast<uint32_t>(materialComponent.normal),
            static_cast<uint32_t>(materialComponent.metallicRoughness),
            static_cast<uint32_t>(_shadowHandle)} };

      vkCmdPushConstants(commandBuffer, _graphicsPipeline->getVkPipelineLayout(),
                         VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(pc), &pc);

      const auto &meshComponent =
          _registry.getComponent<MeshComponent>(object->getEntity());
      const Buffer &indexBuffer = meshComponent.indexBuffer;
      const Buffer &vertexBuffer = meshComponent.vertexBuffer;
      static constexpr VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.getVkBuffer(),
                             offsets);
      vkCmdBindIndexBuffer(commandBuffer, indexBuffer.getVkBuffer(), 0,
                           meshComponent.indexType);
      vkCmdDrawIndexed(commandBuffer,
                       indexBuffer.getSize() /
                           getIndexSize(meshComponent.indexType),
                       1, 0, 0, 0);
    }

    static constexpr OctreeNode::Subvolume options[] = {
        OctreeNode::Subvolume::LOWER_LEFT_BACK,
        OctreeNode::Subvolume::LOWER_LEFT_FRONT,
        OctreeNode::Subvolume::LOWER_RIGHT_BACK,
        OctreeNode::Subvolume::LOWER_RIGHT_FRONT,
        OctreeNode::Subvolume::UPPER_LEFT_BACK,
        OctreeNode::Subvolume::UPPER_LEFT_FRONT,
        OctreeNode::Subvolume::UPPER_RIGHT_BACK,
        OctreeNode::Subvolume::UPPER_RIGHT_FRONT};

    for (OctreeNode::Subvolume option : options) {
      const OctreeNode *childNode = node->getChild(option);
      if (childNode && childNode->getVolume().intersectsFrustum(planes)) {
        nodeQueue.push(childNode);
      }
    }
  }
}

void Application::recordCommandBuffer(uint32_t imageIndex) {
  const Framebuffer &framebuffer = _framebuffers[imageIndex];
  const CommandBuffer &primaryCommandBuffer =
      _primaryCommandBuffer[_currentFrame];
  primaryCommandBuffer.beginAsPrimary();
  primaryCommandBuffer.beginRenderPass(framebuffer);

  static const bool viewportScissorInheritance =
      _physicalDevice->hasAvailableExtension(
          VK_NV_INHERITED_VIEWPORT_SCISSOR_EXTENSION_NAME);

  VkCommandBufferInheritanceViewportScissorInfoNV scissorViewportInheritance;
  if (viewportScissorInheritance) [[likely]] {
    scissorViewportInheritance = VkCommandBufferInheritanceViewportScissorInfoNV{
        .sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO_NV,
        .viewportScissor2D = VK_TRUE,
        .viewportDepthCount = 1,
        .pViewportDepths = &framebuffer.getViewport(),
    };
  }

  std::future<void> futures[MAX_THREADS_IN_POOL];

  futures[0] = std::async(std::launch::async, [&]() -> void {
    const VkCommandBuffer commandBuffer =
        _secondaryCommandBuffers[0][_currentFrame].getVkCommandBuffer();

    if (viewportScissorInheritance) [[likely]] {

      _secondaryCommandBuffers[0][_currentFrame].beginAsSecondary(
          framebuffer, &scissorViewportInheritance);
    } else {

      _secondaryCommandBuffers[0][_currentFrame].beginAsSecondary(framebuffer,
                                                                  nullptr);
      vkCmdSetViewport(commandBuffer, 0, 1, &framebuffer.getViewport());
      vkCmdSetScissor(commandBuffer, 0, 1, &framebuffer.getScissor());
    }
    vkCmdBindPipeline(commandBuffer, _graphicsPipeline->getVkPipelineBindPoint(),
                      _graphicsPipeline->getVkPipeline());

    const OctreeNode *root = _octree->getRoot();
    const auto &planes = extractFrustumPlanes(_camera.getProjectionMatrix() *
                                              _camera.getViewMatrix());

    VkDescriptorSet descriptorSets[] = {
        _bindlessDescriptorSet.getVkDescriptorSet(),
        _dynamicDescriptorSet.getVkDescriptorSet()};

    uint32_t offset;

    _dynamicDescriptorSetWriter.getDynamicBufferSizesWithOffsets(
        &offset, {_currentFrame});

    vkCmdBindDescriptorSets(commandBuffer,
                            _graphicsPipeline->getVkPipelineBindPoint(),
                            _graphicsPipeline->getVkPipelineLayout(), 0,
                            static_cast<uint32_t>(std::size(descriptorSets)),
                            descriptorSets, 1, &offset);

    recordOctreeSecondaryCommandBuffer(commandBuffer, root, planes);

    CHECK_VKCMD(vkEndCommandBuffer(commandBuffer),
                "Failed to vkEndCommandBuffer.");
  });

  futures[1] = std::async(std::launch::async, [&]() -> void {
    // Skybox
    const VkCommandBuffer commandBuffer =
        _secondaryCommandBuffers[1][_currentFrame].getVkCommandBuffer();

    if (viewportScissorInheritance) [[likely]] {

      _secondaryCommandBuffers[1][_currentFrame].beginAsSecondary(
          framebuffer, &scissorViewportInheritance);
    } else {

      _secondaryCommandBuffers[1][_currentFrame].beginAsSecondary(framebuffer,
                                                                  nullptr);
      vkCmdSetViewport(commandBuffer, 0, 1, &framebuffer.getViewport());
      vkCmdSetScissor(commandBuffer, 0, 1, &framebuffer.getScissor());
    }

    vkCmdBindPipeline(commandBuffer, _skyboxPipeline->getVkPipelineBindPoint(),
                      _skyboxPipeline->getVkPipeline());

    static constexpr VkDeviceSize offsets[] = {0};

    vkCmdBindVertexBuffers(commandBuffer, 0, 1,
                           &_vertexBufferCube.getVkBuffer(), offsets);

    vkCmdBindIndexBuffer(commandBuffer, _indexBufferCube.getVkBuffer(), 0,
                         _indexBufferCubeType);

    const PushConstantsSkybox pc = {
        .proj = _camera.getProjectionMatrix(),
        .view = _camera.getViewMatrix(),
        .skyboxHandle = static_cast<uint32_t>(_envMappingTextureHandle)};
    vkCmdPushConstants(commandBuffer, _skyboxPipeline->getVkPipelineLayout(),
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);

    const VkDescriptorSet descriptorSets[] = {
        _bindlessDescriptorSet.getVkDescriptorSet(),
        _dynamicDescriptorSet.getVkDescriptorSet() };

    vkCmdBindDescriptorSets(commandBuffer,
                            _skyboxPipeline->getVkPipelineBindPoint(),
                            _skyboxPipeline->getVkPipelineLayout(), 0, 1,
                            descriptorSets, 0, nullptr);

    vkCmdDrawIndexed(commandBuffer,
                     _indexBufferCube.getSize() /
                         getIndexSize(_indexBufferCubeType),
                     1, 0, 0, 0);

    

    // Env mapping
    vkCmdBindPipeline(commandBuffer, _phongEnvMappingPipeline->getVkPipelineBindPoint(),
        _phongEnvMappingPipeline->getVkPipeline());

    uint32_t offset;

    _dynamicDescriptorSetWriter.getDynamicBufferSizesWithOffsets(
        &offset, { _currentFrame });

    vkCmdBindDescriptorSets(commandBuffer,
        _phongEnvMappingPipeline->getVkPipelineBindPoint(),
        _phongEnvMappingPipeline->getVkPipelineLayout(), 0, std::size(descriptorSets),
        descriptorSets, 1, &offset);

    const PushConstantsModelDescriptorHandles envMapPc = {
        .model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.0f, 0.0f)),
        .descriptorHandles = {
            static_cast<uint32_t>(_envMappingHandle),
            static_cast<uint32_t>(_lightHandle)} };

    vkCmdPushConstants(commandBuffer, _phongEnvMappingPipeline->getVkPipelineLayout(),
        VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(envMapPc), &envMapPc);

    vkCmdBindVertexBuffers(commandBuffer, 0, 1,
        &_vertexBufferCubeNormals.getVkBuffer(), offsets);

    vkCmdBindIndexBuffer(commandBuffer, _indexBufferCube.getVkBuffer(), 0,
        _indexBufferCubeType);

    vkCmdDrawIndexed(commandBuffer,
        _indexBufferCube.getSize() /
        getIndexSize(_indexBufferCubeType),
        1, 0, 0, 0);

    CHECK_VKCMD(vkEndCommandBuffer(commandBuffer),
                "Failed to vkEndCommandBuffer.");
  });

  std::for_each(std::begin(futures), std::end(futures),
                [](std::future<void> &future) { future.wait(); });

  primaryCommandBuffer.executeSecondaryCommandBuffers(
      {_secondaryCommandBuffers[0][_currentFrame].getVkCommandBuffer(),
       _secondaryCommandBuffers[1][_currentFrame].getVkCommandBuffer()});
  primaryCommandBuffer.endRenderPass();

  if (primaryCommandBuffer.end() != VK_SUCCESS) {
    throw std::runtime_error("failed to record command buffer!");
  }
}

void Application::recordShadowCommandBuffer(VkCommandBuffer commandBuffer) {
  const VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

  VkExtent2D extent = _shadowMap.getVkExtent2D();

  std::span<const VkClearValue> clearValues =
      _shadowRenderPass.getAttachmentsLayout().getVkClearValues();

  const VkRenderPassBeginInfo renderPassInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = _shadowRenderPass.getVkRenderPass(),
      .framebuffer = _shadowFramebuffer.getVkFramebuffer(),
      .renderArea = {.offset = {0, 0}, .extent = extent},
      .clearValueCount = static_cast<uint32_t>(clearValues.size()),
      .pClearValues = clearValues.data()};

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  const VkViewport viewport = {.x = 0.0f,
                               .y = 0.0f,
                               .width = static_cast<float>(extent.width),
                               .height = static_cast<float>(extent.height),
                               .minDepth = 0.0f,
                               .maxDepth = 1.0f};
  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

  const VkRect2D scissor = {.offset = {0, 0}, .extent = extent};
  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

  const VkDeviceSize offsets[] = {0};
  vkCmdBindPipeline(commandBuffer, _shadowPipeline->getVkPipelineBindPoint(),
                    _shadowPipeline->getVkPipeline());

  PushConstantsShadow pc = {.lightProjView = _ubLight.projView};

  for (const Object &object : _objects) {
    const auto &meshComponent =
        _registry.getComponent<MeshComponent>(object.getEntity());
    const auto &transformComponent =
        _registry.getComponent<TransformComponent>(object.getEntity());

    pc.model = transformComponent.model;

    vkCmdPushConstants(commandBuffer, _shadowPipeline->getVkPipelineLayout(),
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

    VkBuffer vertexBuffer = meshComponent.vertexBufferPrimitive.getVkBuffer();
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);

    const Buffer &indexBuffer = meshComponent.indexBuffer;
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer.getVkBuffer(), 0,
                         meshComponent.indexType);

    vkCmdDrawIndexed(commandBuffer,
                     indexBuffer.getSize() /
                         getIndexSize(meshComponent.indexType),
                     1, 0, 0, 0);
  }

  vkCmdEndRenderPass(commandBuffer);
}

void Application::recordEnvMappingCommandBuffer(VkCommandBuffer commandBuffer) {
  const VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

  VkExtent2D extent = _envMappingAttachments[0].getVkExtent2D();

  std::span<const VkClearValue> clearValues =
      _envMappingRenderPass.getAttachmentsLayout().getVkClearValues();

  const VkRenderPassBeginInfo renderPassInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = _envMappingRenderPass.getVkRenderPass(),
      .framebuffer = _envMappingFramebuffer.getVkFramebuffer(),
      .renderArea = {.offset = {0, 0}, .extent = extent},
      .clearValueCount = static_cast<uint32_t>(clearValues.size()),
      .pClearValues = clearValues.data()};

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  const VkViewport viewport = {.x = 0.0f,
                               .y = 0.0f,
                               .width = static_cast<float>(extent.width),
                               .height = static_cast<float>(extent.height),
                               .minDepth = 0.0f,
                               .maxDepth = 1.0f};
  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

  const VkRect2D scissor = {.offset = {0, 0}, .extent = extent};
  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

  const VkDescriptorSet descriptorSets[] = {
      _bindlessDescriptorSet.getVkDescriptorSet()};

  vkCmdBindPipeline(commandBuffer,
                    _envMappingPipeline->getVkPipelineBindPoint(),
                    _envMappingPipeline->getVkPipeline());

  vkCmdBindDescriptorSets(commandBuffer,
                          _envMappingPipeline->getVkPipelineBindPoint(),
                          _envMappingPipeline->getVkPipelineLayout(), 0, 1,
                          descriptorSets, 0, nullptr);

  const VkDeviceSize offsets[] = {0};

  for (const Object &object : _objects) {
    const auto &meshComponent =
        _registry.getComponent<MeshComponent>(object.getEntity());
    const auto &transformComponent =
        _registry.getComponent<TransformComponent>(object.getEntity());
    const auto &materialComponent =
        _registry.getComponent<MaterialComponent>(object.getEntity());

    const PushConstantsModelDescriptorHandles pc = {
        .model = transformComponent.model,
        .descriptorHandles = {
            static_cast<uint32_t>(_envMappingHandle),
            static_cast<uint32_t>(materialComponent.diffuse),
            static_cast<uint32_t>(materialComponent.normal),
            static_cast<uint32_t>(materialComponent.metallicRoughness),
            static_cast<uint32_t>(_shadowHandle)}};

    vkCmdPushConstants(
        commandBuffer, _envMappingPipeline->getVkPipelineLayout(),
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        sizeof(pc), &pc);

    VkBuffer vertexBuffer = meshComponent.vertexBuffer.getVkBuffer();
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);

    const Buffer &indexBuffer = meshComponent.indexBuffer;
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer.getVkBuffer(), 0,
                         meshComponent.indexType);

    vkCmdDrawIndexed(commandBuffer,
                     indexBuffer.getSize() /
                         getIndexSize(meshComponent.indexType),
                     1, 0, 0, 0);
  }

  vkCmdEndRenderPass(commandBuffer);
}

void Application::recreateSwapChain() {
  Extent2D extent{};
  while (extent.width == 0 || extent.height == 0) {
    extent = _window->getFramebufferSize();
  }

  Projection oldProjection = _camera.getProjection();
  if (auto projection = std::get_if<PerspectiveProjection>(&oldProjection);
      projection != nullptr) {
    projection->aspect = static_cast<float>(extent.width) / extent.height;
    _camera.setProjection(*projection);
  }
  vkDeviceWaitIdle(_logicalDevice.getVkDevice());

  _swapchain = SwapchainBuilder()
                   .withOldSwapchain(_swapchain.getVkSwapchain())
                   .build(_logicalDevice, _surface.getVkSurface(),
                          VkExtent2D{extent.width, extent.height});
  _attachments.clear();
  _framebuffers.clear();

  {
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    const VkCommandBuffer commandBuffer = handle.getCommandBuffer();
    for (uint8_t i = 0; i < _swapchain.getImagesCount(); ++i) {
      Framebuffer framebuffer = Framebuffer::createFromSwapchain(
          commandBuffer, _renderPass, _swapchain.getExtent(),
          _swapchain.getSwapchainVkImageView(i), _attachments);
      _framebuffers.push_back(std::move(framebuffer));
    }
  }
}