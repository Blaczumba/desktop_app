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
#include "bejzak_engine/common/status/status.h"
#include "bejzak_engine/common/util/vertex_builder.h"
#include "bejzak_engine/common/window/window_glfw.h"
#include "bejzak_engine/lib/buffer/shared_buffer.h"
#include "bejzak_engine/vulkan_wrapper/pipeline/shader_program.h"
#include "bejzak_engine/vulkan_wrapper/render_pass/attachment_layout.h"
#include "bejzak_engine/vulkan_wrapper/util/check.h"

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

ErrorOr<Texture> createSkybox(const LogicalDevice &logicalDevice,
                              VkCommandBuffer commandBuffer,
                              const AssetManager::ImageData &imageData,
                              VkFormat format, float samplerAnisotropy) {
  ASSIGN_OR_RETURN(
      Texture texture,
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
                      createBufferImageCopyRegions(imageData.copyRegions)));
  RETURN_IF_ERROR(texture.addCreateVkImageView(0, imageData.mipLevels, 0, 6));
  return texture;
}

ErrorOr<Texture> createCubemap(const LogicalDevice &logicalDevice,
                               VkCommandBuffer commandBuffer,
                               VkImageAspectFlags aspect, VkFormat format,
                               VkImageUsageFlags additionalUsage,
                               float samplerAnisotropy) {
  ASSIGN_OR_RETURN(
      Texture texture,
      TextureBuilder()
          .withAspect(aspect)
          .withExtent(1024, 1024)
          .withFormat(format)
          .withUsage(VK_IMAGE_USAGE_SAMPLED_BIT | additionalUsage)
          .withLayerCount(6)
          .withAdditionalCreateInfoFlags(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
          .withMaxAnisotropy(samplerAnisotropy)
          .withNumSamples(VK_SAMPLE_COUNT_1_BIT)
          .withMipmapMode(VK_SAMPLER_MIPMAP_MODE_NEAREST)
          .buildAttachment(logicalDevice, commandBuffer));
  RETURN_IF_ERROR(texture.addCreateVkImageView(0, 1, 0, 6));
  return texture;
}

ErrorOr<Texture> createShadowmap(const LogicalDevice &logicalDevice,
                                 VkCommandBuffer commandBuffer, uint32_t width,
                                 uint32_t height, VkFormat format) {
  ASSIGN_OR_RETURN(
      Texture texture,
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
          .buildImageSampler(logicalDevice, commandBuffer));
  RETURN_IF_ERROR(texture.addCreateVkImageView(0, 1, 0, 1));
  return texture;
}

ErrorOr<Texture> createTexture2D(const LogicalDevice &logicalDevice,
                                 VkCommandBuffer commandBuffer,
                                 const AssetManager::ImageData &imageData,
                                 VkFormat format, float samplerAnisotropy) {
  ASSIGN_OR_RETURN(Texture texture,
                   TextureBuilder()
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
                                             imageData.copyRegions)));
  RETURN_IF_ERROR(texture.addCreateVkImageView(0, imageData.mipLevels, 0, 1));
  return texture;
}

constexpr std::string_view engineErrorToString(EngineError error) {
  switch (error) {
  case EngineError::INDEX_OUT_OF_RANGE:
    return "Index out of range";
  case EngineError::EMPTY_COLLECTION:
    return "Empty collection";
  case EngineError::SIZE_MISMATCH:
    return "Size mismatch";
  case EngineError::NOT_RECOGNIZED_TYPE:
    return "Not recognized type";
  case EngineError::NOT_FOUND:
    return "Not found";
  case EngineError::NOT_MAPPED:
    return "Not mapped";
  case EngineError::RESOURCE_EXHAUSTED:
    return "Resource exhausted";
  case EngineError::LOAD_FAILURE:
    return "Load failure";
  case EngineError::FLAG_NOT_SPECIFIED:
    return "Flag not specified";
  }
  return "Unknown EngineError";
}

constexpr std::string_view vkResultToString(int result) {
  switch (result) {
  case VK_SUCCESS:
    return "VK_SUCCESS";
  case VK_NOT_READY:
    return "VK_NOT_READY";
  case VK_TIMEOUT:
    return "VK_TIMEOUT";
  case VK_EVENT_SET:
    return "VK_EVENT_SET";
  case VK_EVENT_RESET:
    return "VK_EVENT_RESET";
  case VK_INCOMPLETE:
    return "VK_INCOMPLETE";
  case VK_ERROR_OUT_OF_HOST_MEMORY:
    return "VK_ERROR_OUT_OF_HOST_MEMORY";
  case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
  case VK_ERROR_INITIALIZATION_FAILED:
    return "VK_ERROR_INITIALIZATION_FAILED";
  case VK_ERROR_DEVICE_LOST:
    return "VK_ERROR_DEVICE_LOST";
  case VK_ERROR_MEMORY_MAP_FAILED:
    return "VK_ERROR_MEMORY_MAP_FAILED";
  case VK_ERROR_LAYER_NOT_PRESENT:
    return "VK_ERROR_LAYER_NOT_PRESENT";
  case VK_ERROR_EXTENSION_NOT_PRESENT:
    return "VK_ERROR_EXTENSION_NOT_PRESENT";
  case VK_ERROR_FEATURE_NOT_PRESENT:
    return "VK_ERROR_FEATURE_NOT_PRESENT";
  case VK_ERROR_INCOMPATIBLE_DRIVER:
    return "VK_ERROR_INCOMPATIBLE_DRIVER";
  case VK_ERROR_TOO_MANY_OBJECTS:
    return "VK_ERROR_TOO_MANY_OBJECTS";
  case VK_ERROR_FORMAT_NOT_SUPPORTED:
    return "VK_ERROR_FORMAT_NOT_SUPPORTED";
  }
  return "Unknown VkResult error code";
}

std::string_view errorToString(const ErrorType &error) {
  if (std::holds_alternative<int>(error)) {
    return vkResultToString(std::get<int>(error));
  } else {
    return engineErrorToString(std::get<EngineError>(error));
  }
}

} // namespace

Application::Application(const std::shared_ptr<FileLoader> &fileLoader)
    : _camera(PerspectiveProjection{glm::radians(45.0f), 1920.0f / 1080.f,
                                    0.01f, 50.0f},
              glm::vec3(0.0f), 5.5f, 0.01f),
      _programManager(fileLoader) {
  if (Status status = init(); !status) {
    std::println("Failed to initialize application: {}",
                 errorToString(status.error()));
  }

  _assetManager = AssetManager(_logicalDevice, fileLoader);

  if (Status status = loadCubemap(); !status) {
    std::println("Failed to load cubemap: {}", errorToString(status.error()));
  }

  if (Status status = createDescriptorSets(); !status) {
    std::println("Failed to create descriptor sets: {}",
                 errorToString(status.error()));
  }

  if (Status status = loadObjects(); !status) {
    std::println("Failed to load objects: {}", errorToString(status.error()));
  }

  if (Status status = createOctreeScene(); !status) {
    std::println("Failed to create octree scene: {}",
                 errorToString(status.error()));
  }

  if (Status status = createPresentResources(); !status) {
    std::println("Failed to create present resources: {}",
                 errorToString(status.error()));
  }

  if (Status status = createShadowResources(); !status) {
    std::println("Failed to create shadow resources: {}",
                 errorToString(status.error()));
  }

  if (Status status = createCommandBuffers(); !status) {
    std::println("Failed to create command buffers: {}",
                 errorToString(status.error()));
  }

  if (Status status = createSyncObjects(); !status) {
    std::println("Failed to create sync objects: {}",
                 errorToString(status.error()));
  }

  if (Status status = createMirrorCubemap(); !status) {
    std::println("Failed to create mirror cubemap: {}",
                 errorToString(status.error()));
  }
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

Status Application::init() {
  _window = std::make_unique<WindowGlfw>("Bejzak Engine", 1920, 1080);
  _mouseKeyboardManager = _window->createMouseKeyboardManager();
  std::vector<const char *> requiredExtensions = _window->getVulkanExtensions();
#ifdef VALIDATION_LAYERS_ENABLED
  requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif // VALIDATION_LAYERS_ENABLED
  requiredExtensions.push_back(
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  ASSIGN_OR_RETURN(
      _instance,
      Instance::create("Bejzak Engine", requiredExtensions, debugCallback));
#ifdef VALIDATION_LAYERS_ENABLED
  ASSIGN_OR_RETURN(_debugMessenger,
                   DebugMessenger::create(_instance, debugCallback));
#endif // VALIDATION_LAYERS_ENABLED

  ASSIGN_OR_RETURN(_surface, Surface::create(_instance, *_window));
  ASSIGN_OR_RETURN(_physicalDevice,
                   PhysicalDevice::create(_instance, _surface.getVkSurface()));
  ASSIGN_OR_RETURN(_logicalDevice, LogicalDevice::create(*_physicalDevice));
  const Extent2D framebufferSize = _window->getFramebufferSize();
  ASSIGN_OR_RETURN(
      _swapchain,
      SwapchainBuilder()
          .withPreferredPresentMode(VK_PRESENT_MODE_MAILBOX_KHR)
          .build(_logicalDevice, _surface.getVkSurface(),
                 VkExtent2D{framebufferSize.width, framebufferSize.height}));
  ASSIGN_OR_RETURN(_singleTimeCommandPool, CommandPool::create(_logicalDevice));
  return StatusOk();
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

Status Application::createMirrorCubemap() {
  const float samplerAnisotropy = _physicalDevice->getMaxSamplerAnisotropy();
  {
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    ASSIGN_OR_RETURN(
        _mirrorCubemapAttachments[0],
        createCubemap(_logicalDevice, handle.getCommandBuffer(),
                      VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R8G8B8A8_SRGB,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, samplerAnisotropy));
    ASSIGN_OR_RETURN(_mirrorCubemapAttachments[1],
                     createCubemap(_logicalDevice, handle.getCommandBuffer(),
                                   VK_IMAGE_ASPECT_DEPTH_BIT,
                                   VK_FORMAT_D24_UNORM_S8_UINT,
                                   VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                   samplerAnisotropy));
  }

  AttachmentLayout attachmentLayout;
  attachmentLayout.addColorAttachment(VK_FORMAT_R8G8B8A8_SRGB,
                                      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                      VK_ATTACHMENT_STORE_OP_STORE);
  attachmentLayout.addDepthAttachment(
      VK_FORMAT_D24_UNORM_S8_UINT, VK_ATTACHMENT_STORE_OP_DONT_CARE);

  ASSIGN_OR_RETURN(
      _mirrorCubemapRenderPass,
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
          .addSubpass({ 0, 1 })
          .build(_logicalDevice));

  ASSIGN_OR_RETURN(_mirrorCubemapFramebuffer,
                   Framebuffer::createFromTextures(_mirrorCubemapRenderPass,
                                                   _mirrorCubemapAttachments));

  ASSIGN_OR_RETURN(_mirrorCubemapShaderProgram, _programManager.createPbrEnvMappingProgram(_logicalDevice));

  float arrayLayer;
  const GraphicsPipelineParameters pipelineParams{
	  .cullMode = VK_CULL_MODE_FRONT_BIT,
      .specializationData = SpecializationData{
          .data = &arrayLayer,
		  .dataSize = sizeof(arrayLayer),
          .mapEntries = {
              {VK_SHADER_STAGE_VERTEX_BIT, {VkSpecializationMapEntry{.constantID = 0, .offset = 0, .size = sizeof(arrayLayer)}}}
          }
      }
  };
  for (int i = 0; i < 6; ++i) {
      arrayLayer = static_cast<float>(i);
      _mirrorCubemapPipeline[i] = std::make_unique<GraphicsPipeline>(
              _mirrorCubemapRenderPass, _mirrorCubemapShaderProgram, pipelineParams);
  }

  // _mirrorCubemapTextureHandle = _bindlessWriter->storeTexture(_mirrorCubemapAttachments[0]);
  const glm::vec3 pos = glm::vec3(0.0f, 2.0f, 0.0f);
  glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 50.0f);
  // proj[1][1] *= -1; // Invert Y for Vulkan

  struct {
    alignas(16) glm::mat4 projView[6];
    alignas(16) glm::vec3 viewPos;
    alignas(16) glm::mat4 lightProjView;
    alignas(16) glm::vec3 lightPos;
  } const faceTransform = {
    .projView = {
      proj * glm::lookAt(pos, pos + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
      proj * glm::lookAt(pos, pos + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
      proj * glm::lookAt(pos, pos + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
      proj * glm::lookAt(pos, pos + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
      proj * glm::lookAt(pos, pos + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
      proj * glm::lookAt(pos, pos + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
    },
	.viewPos = pos,
	.lightProjView = _ubLight.projView,
	.lightPos = _ubLight.pos
  };

  ASSIGN_OR_RETURN(_mirrorCubemapUniformBuffer, Buffer::createUniformBuffer(_logicalDevice, sizeof(faceTransform)));
  RETURN_IF_ERROR(_mirrorCubemapUniformBuffer.copyData(faceTransform));
  _mirrorCubemapHandle = _bindlessWriter->storeBuffer(_mirrorCubemapUniformBuffer);
  _mirrorCubemapTextureHandle = _bindlessWriter->storeTexture(_mirrorCubemapAttachments[0]);

  return StatusOk();
}

Status Application::loadCubemap() {
  _assetManager.loadImageAsync(TEXTURES_PATH "cubemap_yokohama_rgba.ktx");
  // TODO: temporal experiment
  auto fileLoader = std::make_unique<StandardFileLoader>();
  ASSIGN_OR_RETURN(std::string data,
                   fileLoader->loadFileToString(MODELS_PATH "cube.obj"));
  ASSIGN_OR_RETURN(const VertexData vertexDataCube,
                   loadObj(_assetManager, "cube.obj", data));

  {
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    const VkCommandBuffer commandBuffer = handle.getCommandBuffer();

    ASSIGN_OR_RETURN(
        const AssetManager::ImageData &imageData,
        _assetManager.getImageData(TEXTURES_PATH "cubemap_yokohama_rgba.ktx"));

    ASSIGN_OR_RETURN(_textureCubemap,
                     createSkybox(_logicalDevice, commandBuffer, imageData,
                                  VK_FORMAT_R8G8B8A8_UNORM,
                                  _physicalDevice->getMaxSamplerAnisotropy()));

    ASSIGN_OR_RETURN(const AssetManager::VertexData &vData,
                     _assetManager.getVertexData("cube.obj"));
    ASSIGN_OR_RETURN(
        _vertexBufferCube,
        Buffer::createVertexBuffer(_logicalDevice,
                                   vData.vertexBufferPositions.getSize()));
    RETURN_IF_ERROR(_vertexBufferCube.copyBuffer(commandBuffer,
                                                 vData.vertexBufferPositions));
    ASSIGN_OR_RETURN(
        _indexBufferCube,
        Buffer::createIndexBuffer(_logicalDevice, vData.indexBuffer.getSize()));
    RETURN_IF_ERROR(
        _indexBufferCube.copyBuffer(commandBuffer, vData.indexBuffer));
    _indexBufferCubeType = vData.indexType;
  }

  return StatusOk();
}

Status Application::loadObjects() {
  // TODO needs refactoring
  ASSIGN_OR_RETURN(
      const std::vector<VertexData> sceneData,
      LoadGltfFromFile(_assetManager, MODELS_PATH "sponza/scene.gltf"));
  const float maxSamplerAnisotropy = _physicalDevice->getMaxSamplerAnisotropy();
  _objects.reserve(sceneData.size());

  SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
  const VkCommandBuffer commandBuffer = handle.getCommandBuffer();
  for (const VertexData &sceneObject : sceneData) {
    const std::string diffusePath =
        MODELS_PATH "sponza/" + sceneObject.diffuseTexture;
    if (!_textures.contains(diffusePath)) {
      ASSIGN_OR_RETURN(const AssetManager::ImageData &imgData,
                       _assetManager.getImageData(diffusePath));
      ASSIGN_OR_RETURN(Texture texture,
                       createTexture2D(_logicalDevice, commandBuffer, imgData,
                                       VK_FORMAT_R8G8B8A8_SRGB,
                                       maxSamplerAnisotropy));
      _textures.emplace(diffusePath,
                        std::make_pair(_bindlessWriter->storeTexture(texture),
                                       std::move(texture)));
    }

    const std::string normalPath =
        MODELS_PATH "sponza/" + sceneObject.normalTexture;
    if (!_textures.contains(normalPath)) {
      ASSIGN_OR_RETURN(const AssetManager::ImageData &imgData,
                       _assetManager.getImageData(normalPath));
      ASSIGN_OR_RETURN(Texture texture,
                       createTexture2D(_logicalDevice, commandBuffer, imgData,
                                       VK_FORMAT_R8G8B8A8_UNORM,
                                       maxSamplerAnisotropy));
      _textures.emplace(normalPath,
                        std::make_pair(_bindlessWriter->storeTexture(texture),
                                       std::move(texture)));
    }

    const std::string metallicRoughnessPath =
        MODELS_PATH "sponza/" + sceneObject.metallicRoughnessTexture;
    if (!_textures.contains(metallicRoughnessPath)) {
      ASSIGN_OR_RETURN(const AssetManager::ImageData &imgData,
                       _assetManager.getImageData(metallicRoughnessPath));
      ASSIGN_OR_RETURN(Texture texture,
                       createTexture2D(_logicalDevice, commandBuffer, imgData,
                                       VK_FORMAT_R8G8B8A8_UNORM,
                                       maxSamplerAnisotropy));
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
    ASSIGN_OR_RETURN(const AssetManager::VertexData &vData,
                     _assetManager.getVertexData(sceneObject.vertexResource));
    MeshComponent msh;
    ASSIGN_OR_RETURN(msh.vertexBuffer,
                     Buffer::createVertexBuffer(_logicalDevice,
                                                vData.vertexBuffer.getSize()));
    RETURN_IF_ERROR(
        msh.vertexBuffer.copyBuffer(commandBuffer, vData.vertexBuffer));
    ASSIGN_OR_RETURN(
        msh.indexBuffer,
        Buffer::createIndexBuffer(_logicalDevice, vData.indexBuffer.getSize()));
    RETURN_IF_ERROR(
        msh.indexBuffer.copyBuffer(commandBuffer, vData.indexBuffer));
    ASSIGN_OR_RETURN(
        msh.vertexBufferPrimitive,
        Buffer::createVertexBuffer(_logicalDevice,
                                   vData.vertexBufferPositions.getSize()));
    RETURN_IF_ERROR(msh.vertexBufferPrimitive.copyBuffer(
        commandBuffer, vData.vertexBufferPositions));
    msh.indexType = vData.indexType;
    msh.aabb = createAABBfromVertices(sceneObject.positions, sceneObject.model);
    _registry.addComponent<MeshComponent>(e, std::move(msh));

    TransformComponent trsf;
    trsf.model = sceneObject.model;
    _registry.addComponent<TransformComponent>(e, std::move(trsf));
  }

  return StatusOk();
}

Status Application::createOctreeScene() {
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

  return StatusOk();
}

Status Application::createDescriptorSets() {
  const uint32_t size = _logicalDevice.getPhysicalDevice().getMemoryAlignment(
      sizeof(UniformBufferCamera));
  ASSIGN_OR_RETURN(
      _dynamicUniformBuffersCamera,
      Buffer::createUniformBuffer(_logicalDevice, MAX_FRAMES_IN_FLIGHT * size));

  ASSIGN_OR_RETURN(_pbrShaderProgram,
                   _programManager.createPBRProgram(_logicalDevice));
  ASSIGN_OR_RETURN(_skyboxShaderProgram,
                   _programManager.createSkyboxProgram(_logicalDevice));

  ASSIGN_OR_RETURN(
      _descriptorPool,
      DescriptorPool::create(_logicalDevice, 150,
                             VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT));
  ASSIGN_OR_RETURN(_dynamicDescriptorPool,
                   DescriptorPool::create(_logicalDevice, 1));

  ASSIGN_OR_RETURN(_bindlessDescriptorSet,
                   _descriptorPool->createDesriptorSet(
                       _programManager.getVkDescriptorSetLayout(
                           DescriptorSetType::BINDLESS)));
  ASSIGN_OR_RETURN(
      _dynamicDescriptorSet,
      _dynamicDescriptorPool->createDesriptorSet(
          _programManager.getVkDescriptorSetLayout(DescriptorSetType::CAMERA)));
  _bindlessWriter =
      std::make_unique<BindlessDescriptorSetWriter>(_bindlessDescriptorSet);
  _skyboxHandle = _bindlessWriter->storeTexture(_textureCubemap);

  _dynamicDescriptorSetWriter.storeDynamicBuffer(_dynamicUniformBuffersCamera,
                                                 size);
  _dynamicDescriptorSetWriter.writeDescriptorSet(
      _logicalDevice.getVkDevice(), _dynamicDescriptorSet.getVkDescriptorSet());

  ASSIGN_OR_RETURN(
      _lightBuffer,
      Buffer::createUniformBuffer(_logicalDevice, sizeof(UniformBufferLight)));
  _lightHandle = _bindlessWriter->storeBuffer(_lightBuffer);

  _ubLight.pos = glm::vec3(15.1891f, 2.66408f, -0.841221f);
  _ubLight.projView = glm::perspective(glm::radians(120.0f), 1.0f, 0.1f, 40.0f);
  _ubLight.projView[1][1] = -_ubLight.projView[1][1];
  _ubLight.projView =
      _ubLight.projView * glm::lookAt(_ubLight.pos,
                                      glm::vec3(-3.82383f, 3.66503f, 1.30751f),
                                      glm::vec3(0.0f, 1.0f, 0.0f));
  _lightBuffer.copyData(_ubLight, 0);

  return StatusOk();
}

Status Application::createPresentResources() {
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

  ASSIGN_OR_RETURN(
      _renderPass,
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
          .build(_logicalDevice));

  {
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    const VkCommandBuffer commandBuffer = handle.getCommandBuffer();
    for (uint8_t i = 0; i < _swapchain.getImagesCount(); ++i) {
      ASSIGN_OR_RETURN(auto framebuffer,
                       Framebuffer::createFromSwapchain(
                           commandBuffer, _renderPass, _swapchain.getExtent(),
                           _swapchain.getSwapchainVkImageView(i),
                           _attachments));
      _framebuffers.push_back(std::move(framebuffer));
    }
  }
  const GraphicsPipelineParameters pbrPipelineParameters = {
      .msaaSamples = msaaSamples,
      // .patchControlPoints = 3,
  };
  _graphicsPipeline = std::make_unique<GraphicsPipeline>(
      _renderPass, _pbrShaderProgram, pbrPipelineParameters);
  const GraphicsPipelineParameters skyboxPipelineParameters = {
      .cullMode = VK_CULL_MODE_FRONT_BIT, .msaaSamples = msaaSamples};
  _graphicsPipelineSkybox = std::make_unique<GraphicsPipeline>(
      _renderPass, _skyboxShaderProgram, skyboxPipelineParameters);
  return StatusOk();
}

Status Application::createShadowResources() {
  {
    // TODO: Should not be in this function.
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    const VkCommandBuffer commandBuffer = handle.getCommandBuffer();
    ASSIGN_OR_RETURN(_shadowMap,
                     createShadowmap(_logicalDevice, commandBuffer, 1024 * 2,
                                     1024 * 2, VK_FORMAT_D32_SFLOAT));
  }
  _shadowHandle = _bindlessWriter->storeTexture(_shadowMap);

  ASSIGN_OR_RETURN(_shadowShaderProgram,
                   _programManager.createShadowProgram(_logicalDevice));
  AttachmentLayout attachmentLayout;
  attachmentLayout.addShadowAttachment(
      VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  ASSIGN_OR_RETURN(_shadowRenderPass, RenderpassBuilder(attachmentLayout)
                                          .addSubpass({0})
                                          .build(_logicalDevice));
  ASSIGN_OR_RETURN(_shadowFramebuffer,
                   Framebuffer::createFromTextures(_shadowRenderPass,
                                                   std::span(&_shadowMap, 1)));

  const GraphicsPipelineParameters parameters = {
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .depthBiasConstantFactor = 0.7f,
      .depthBiasSlopeFactor = 2.0f,
  };
  _shadowPipeline = std::make_unique<GraphicsPipeline>(
      _shadowRenderPass, _shadowShaderProgram, parameters);
  return StatusOk();
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
  }
  {
      SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
      recordMirrorCommandBuffer(handle.getCommandBuffer());
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

  _primaryCommandBuffer[_currentFrame].resetCommandBuffer();
  for (int i = 0; i < MAX_THREADS_IN_POOL; i++)
    _commandBuffers[i][_currentFrame].resetCommandBuffer();

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

Status Application::createSyncObjects() {
  static constexpr VkSemaphoreCreateInfo semaphoreInfo = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  static constexpr VkFenceCreateInfo fenceInfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT};

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    CHECK_VKCMD(vkCreateSemaphore(_logicalDevice.getVkDevice(), &semaphoreInfo,
                                  nullptr, &_imageAvailableSemaphores[i]));
    CHECK_VKCMD(vkCreateSemaphore(_logicalDevice.getVkDevice(), &semaphoreInfo,
                                  nullptr, &_renderFinishedSemaphores[i]));
    CHECK_VKCMD(vkCreateFence(_logicalDevice.getVkDevice(), &fenceInfo, nullptr,
                              &_inFlightFences[i]));
  }
  return StatusOk();
}

void Application::updateUniformBuffer(uint32_t currentFrame) {
  _ubCamera.view = _camera.getViewMatrix();
  _ubCamera.proj = _camera.getProjectionMatrix();
  _ubCamera.pos = _camera.getPosition();
  _dynamicUniformBuffersCamera.copyData(
      _ubCamera, currentFrame * _physicalDevice->getMemoryAlignment(
                                    sizeof(UniformBufferCamera)));
}

Status Application::createCommandBuffers() {
  for (int i = 0; i < MAX_THREADS_IN_POOL + 1; i++) {
    ASSIGN_OR_RETURN(_commandPools[i], CommandPool::create(_logicalDevice));
  }
  ASSIGN_OR_RETURN(_primaryCommandBuffer,
                   _commandPools[MAX_THREADS_IN_POOL]
                       ->createPrimaryCommandBuffers<MAX_FRAMES_IN_FLIGHT>());
  for (int i = 0; i < MAX_THREADS_IN_POOL; i++) {
    ASSIGN_OR_RETURN(
        _commandBuffers[i],
        _commandPools[i]
            ->createSecondaryCommandBuffers<MAX_FRAMES_IN_FLIGHT>());
  }
  return StatusOk();
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

      const PushConstantsPBR pc = {
          .model = transformComponent.model,
          .uniformIndex = (uint32_t)_lightHandle,
          .diffuse = (uint32_t)materialComponent.diffuse,
          .normal = (uint32_t)materialComponent.normal,
          .metallicRoughness = (uint32_t)materialComponent.metallicRoughness,
          .shadow = (uint32_t)_shadowHandle,
      };

      vkCmdPushConstants(
          commandBuffer, _graphicsPipeline->getVkPipelineLayout(),
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
          sizeof(pc), &pc);

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
  const PrimaryCommandBuffer &primaryCommandBuffer =
      _primaryCommandBuffer[_currentFrame];
  primaryCommandBuffer.begin();
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

  std::future<Status> futures[MAX_THREADS_IN_POOL];

  futures[0] = std::async(std::launch::async, [&]() -> Status {
    const VkCommandBuffer commandBuffer =
        _commandBuffers[0][_currentFrame].getVkCommandBuffer();

    if (viewportScissorInheritance) [[likely]] {
      CHECK_VKCMD(_commandBuffers[0][_currentFrame].begin(
          framebuffer, &scissorViewportInheritance));
    } else {
      CHECK_VKCMD(
          _commandBuffers[0][_currentFrame].begin(framebuffer, nullptr));
      vkCmdSetViewport(commandBuffer, 0, 1, &framebuffer.getViewport());
      vkCmdSetScissor(commandBuffer, 0, 1, &framebuffer.getScissor());
    }
    vkCmdBindPipeline(commandBuffer,
                      _graphicsPipeline->getVkPipelineBindPoint(),
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

    CHECK_VKCMD(vkEndCommandBuffer(commandBuffer));

    return StatusOk();
  });

  futures[1] = std::async(std::launch::async, [&]() -> Status {
    // Skybox
    const VkCommandBuffer commandBuffer =
        _commandBuffers[1][_currentFrame].getVkCommandBuffer();

    if (viewportScissorInheritance) [[likely]] {
      CHECK_VKCMD(_commandBuffers[1][_currentFrame].begin(
          framebuffer, &scissorViewportInheritance));
    } else {
      CHECK_VKCMD(
          _commandBuffers[1][_currentFrame].begin(framebuffer, nullptr));
      vkCmdSetViewport(commandBuffer, 0, 1, &framebuffer.getViewport());
      vkCmdSetScissor(commandBuffer, 0, 1, &framebuffer.getScissor());
    }

    vkCmdBindPipeline(commandBuffer,
                      _graphicsPipelineSkybox->getVkPipelineBindPoint(),
                      _graphicsPipelineSkybox->getVkPipeline());

    static constexpr VkDeviceSize offsets[] = {0};

    vkCmdBindVertexBuffers(commandBuffer, 0, 1,
                           &_vertexBufferCube.getVkBuffer(), offsets);

    vkCmdBindIndexBuffer(commandBuffer, _indexBufferCube.getVkBuffer(), 0,
                         _indexBufferCubeType);

    const PushConstantsSkybox pc = {.proj = _camera.getProjectionMatrix(),
                                    .view = _camera.getViewMatrix(),
                                    .skyboxHandle =
                                        static_cast<uint32_t>(_mirrorCubemapTextureHandle)};
    vkCmdPushConstants(
        commandBuffer, _graphicsPipelineSkybox->getVkPipelineLayout(),
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        sizeof(pc), &pc);

    const VkDescriptorSet descriptorSet =
        _bindlessDescriptorSet.getVkDescriptorSet();

    vkCmdBindDescriptorSets(commandBuffer,
                            _graphicsPipelineSkybox->getVkPipelineBindPoint(),
                            _graphicsPipelineSkybox->getVkPipelineLayout(), 0,
                            1, &descriptorSet, 0, nullptr);

    vkCmdDrawIndexed(commandBuffer,
                     _indexBufferCube.getSize() /
                         getIndexSize(_indexBufferCubeType),
                     1, 0, 0, 0);

    CHECK_VKCMD(vkEndCommandBuffer(commandBuffer));

    return StatusOk();
  });

  std::for_each(std::begin(futures), std::end(futures),
                [](std::future<Status> &future) { future.wait(); });

  primaryCommandBuffer.executeSecondaryCommandBuffers(
      {_commandBuffers[0][_currentFrame].getVkCommandBuffer(),
       _commandBuffers[1][_currentFrame].getVkCommandBuffer()});
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

void Application::recordMirrorCommandBuffer(VkCommandBuffer commandBuffer) {
    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };

    VkExtent2D extent = _mirrorCubemapAttachments[0].getVkExtent2D();

    std::span<const VkClearValue> clearValues =
        _mirrorCubemapRenderPass.getAttachmentsLayout().getVkClearValues();

    const VkRenderPassBeginInfo renderPassInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = _mirrorCubemapRenderPass.getVkRenderPass(),
        .framebuffer = _mirrorCubemapFramebuffer.getVkFramebuffer(),
        .renderArea = {.offset = {0, 0}, .extent = extent},
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data() };

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
        VK_SUBPASS_CONTENTS_INLINE);

    const VkViewport viewport = { .x = 0.0f,
                                 .y = 0.0f,
                                 .width = static_cast<float>(extent.width),
                                 .height = static_cast<float>(extent.height),
                                 .minDepth = 0.0f,
                                 .maxDepth = 1.0f };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    const VkRect2D scissor = { .offset = {0, 0}, .extent = extent };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    const VkDescriptorSet descriptorSets[] = {
        _bindlessDescriptorSet.getVkDescriptorSet()};

    vkCmdBindDescriptorSets(commandBuffer, _mirrorCubemapPipeline[0]->getVkPipelineBindPoint(),
        _mirrorCubemapPipeline[0]->getVkPipelineLayout(), 0, 1, descriptorSets, 0, nullptr);

    const VkDeviceSize offsets[] = { 0 };

    for (const Object& object : _objects) {
        const auto& meshComponent =
            _registry.getComponent<MeshComponent>(object.getEntity());
        const auto& transformComponent =
            _registry.getComponent<TransformComponent>(object.getEntity());
        const auto& materialComponent =
			_registry.getComponent<MaterialComponent>(object.getEntity());

        const PushConstantsPBR pc = {
          .model = transformComponent.model,
          .uniformIndex = (uint32_t)_mirrorCubemapHandle,
          .diffuse = (uint32_t)materialComponent.diffuse,
          .normal = (uint32_t)materialComponent.normal,
          .metallicRoughness = (uint32_t)materialComponent.metallicRoughness,
          .shadow = (uint32_t)_shadowHandle};

        vkCmdPushConstants(commandBuffer, _mirrorCubemapPipeline[0]->getVkPipelineLayout(),
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

        VkBuffer vertexBuffer = meshComponent.vertexBuffer.getVkBuffer();
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);

        const Buffer& indexBuffer = meshComponent.indexBuffer;
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer.getVkBuffer(), 0,
            meshComponent.indexType);

        for (int i = 0; i < 1; i++) {
            vkCmdBindPipeline(commandBuffer, _mirrorCubemapPipeline[i]->getVkPipelineBindPoint(),
                _mirrorCubemapPipeline[i]->getVkPipeline());
            vkCmdDrawIndexed(commandBuffer,
                indexBuffer.getSize() /
                getIndexSize(meshComponent.indexType),
                1, 0, 0, 0);
        }
    }

    vkCmdEndRenderPass(commandBuffer);
}

Status Application::recreateSwapChain() {
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

  ASSIGN_OR_RETURN(_swapchain,
                   SwapchainBuilder()
                       .withOldSwapchain(_swapchain.getVkSwapchain())
                       .build(_logicalDevice, _surface.getVkSurface(),
                              VkExtent2D{extent.width, extent.height}));
  _attachments.clear();
  _framebuffers.clear();

  {
    SingleTimeCommandBuffer handle(*_singleTimeCommandPool);
    const VkCommandBuffer commandBuffer = handle.getCommandBuffer();
    for (uint8_t i = 0; i < _swapchain.getImagesCount(); ++i) {
      ASSIGN_OR_RETURN(Framebuffer framebuffer,
                       Framebuffer::createFromSwapchain(
                           commandBuffer, _renderPass, _swapchain.getExtent(),
                           _swapchain.getSwapchainVkImageView(i),
                           _attachments));
      _framebuffers.push_back(std::move(framebuffer));
    }
  }

  return StatusOk();
}