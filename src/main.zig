const std = @import("std");
const wio = @import("wio");
const obj = @import("obj");
const vk = @import("vulkan");
const za = @import("zalgebra");
// const tex = @import("mr_texture");
const ktx2 = @import("mr_ktx2");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cInclude("vma/vk_mem_alloc.h");
    @cInclude("ktx.h");
    @cInclude("ktxvulkan.h");
    @cInclude("slang_wrapper.h");
});

const log = std.log.scoped(.htv);
const assert = std.debug.assert;

const max_frames_in_flight = 2;

const suzanne_ktx_bytes = [3][]const u8{
    @embedFile("assets/suzanne0.ktx"),
    @embedFile("assets/suzanne1.ktx"),
    @embedFile("assets/suzanne2.ktx"),
};

const Vertex = extern struct {
    pos: za.Vec3,
    normal: za.Vec3,
    uv: za.Vec2,
};

const ShaderData = extern struct {
    projection: za.Mat4 = za.Mat4.zero(),
    view: za.Mat4 = za.Mat4.zero(),
    model: [3]za.Mat4 = [_]za.Mat4{.zero()} ** 3,
    light_pos: za.Vec4 = za.Vec4.new(0, -10, 10, 0),
    selected: u32 = 1,
};

const ShaderDataBuffer = struct {
    allocation: c.VmaAllocation = undefined,
    allocation_info: c.VmaAllocationInfo = .{},
    buffer: vk.Buffer = .null_handle,
    device_address: vk.DeviceAddress = 0,
};

const Texture = struct {
    allocation: c.VmaAllocation = undefined,
    image: vk.Image = .null_handle,
    view: vk.ImageView = .null_handle,
    sampler: vk.Sampler = .null_handle,
};

var app = struct {
    init: std.process.Init = undefined,
    size: wio.Size = .{ .width = 640, .height = 480 },
    window: wio.Window = undefined,

    vk_base: vk.BaseWrapper = undefined,

    vk_instance_handle: vk.Instance = .null_handle,
    vk_instance_wrapper: vk.InstanceWrapper = undefined,
    instance: vk.InstanceProxy = undefined,

    physical_device: vk.PhysicalDevice = .null_handle,

    surface: vk.SurfaceKHR = .null_handle,
    surface_capabilities: vk.SurfaceCapabilitiesKHR = undefined,
    vk_device_handle: vk.Device = .null_handle,
    vk_device_wrapper: vk.DeviceWrapper = undefined,
    device: vk.DeviceProxy = undefined,

    queue_family_index: u32 = 0,
    queue: vk.Queue = .null_handle,

    vma_allocator: c.VmaAllocator = undefined,

    swapchain: vk.SwapchainKHR = .null_handle,
    swapchain_images: []vk.Image = undefined,
    swapchain_image_views: []vk.ImageView = undefined,

    image_format: vk.Format = .undefined,
    depth_format: vk.Format = .undefined,

    depth_image: vk.Image = .null_handle,
    depth_image_view: vk.ImageView = .null_handle,
    depth_image_allocation: c.VmaAllocation = undefined,

    model_obj: obj.ObjData = undefined,
    vertices: []Vertex = undefined,
    indices: []u16 = undefined,

    vertex_buffer: vk.Buffer = .null_handle,
    vertex_buffer_allocation: c.VmaAllocation = undefined,

    shader_data_buffers: [max_frames_in_flight]ShaderDataBuffer = undefined,

    fences: [max_frames_in_flight]vk.Fence = undefined,
    present_semaphores: [max_frames_in_flight]vk.Semaphore = undefined,
    render_semaphores: []vk.Semaphore = undefined,

    command_pool: vk.CommandPool = .null_handle,
    command_buffers: [max_frames_in_flight]vk.CommandBuffer = undefined,

    textures: [3]Texture = undefined,
    texture_descriptors: [3]vk.DescriptorImageInfo = undefined,

    descriptor_pool: vk.DescriptorPool = .null_handle,
    descriptor_set_layout_tex: vk.DescriptorSetLayout = .null_handle,
    descriptor_set_tex: vk.DescriptorSet = .null_handle,

    slang_wrapper: *c.SlangWrapper = undefined,

    pipeline: vk.Pipeline = .null_handle,
    pipeline_layout: vk.PipelineLayout = .null_handle,

    visible: bool = true,

    frame_index: u32 = 0,
    image_index: u32 = 0,
    update_swapchain: bool = false,

    shader_data: ShaderData = .{},
    cam_pos: za.Vec3 = .new(0, 0, -6),
    object_rotations: [3]za.Vec3 = [_]za.Vec3{.zero()} ** 3,

    rot_speed: f32 = 5.0,
}{};

pub fn main(init: std.process.Init) !void {
    app.init = init;

    try wio.init(init.gpa, init.io, .{});
    defer wio.deinit();

    app.window = try wio.createWindow(.{ .title = "software", .size = app.size, .scale = 1 });
    defer app.window.destroy();

    app.vk_base = .load(@as(*const fn (vk.Instance, [*:0]const u8) ?*const fn () void, @ptrCast(&wio.vkGetInstanceProcAddr)));

    // render splash screen
    var fb = try app.window.createFramebuffer(app.size);
    while (app.window.getEvent()) |event| {
        switch (event) {
            .size_physical => |new_size| {
                if (new_size.width != app.size.width or new_size.height != app.size.height) {
                    fb.destroy();
                    fb = try app.window.createFramebuffer(new_size);
                    app.size = new_size;
                }
            },
            else => {},
        }
    }
    renderSplash(&fb, app.size, 0);
    app.window.presentFramebuffer(&fb);
    fb.destroy();

    // set up vulkan
    try createInstance();
    try selectPhysicalDevice();
    try createSurface();
    try selectQueueFamily();
    try createLogicalDevice();
    try setupVMA();
    try getSurfaceCapabilities();
    try createSwapchain();
    defer init.gpa.free(app.swapchain_images);
    defer init.gpa.free(app.swapchain_image_views);
    try createDepthAttachment();
    try loadMesh();
    defer app.model_obj.deinit(init.gpa);
    defer app.init.gpa.free(app.vertices);
    defer app.init.gpa.free(app.indices);
    try createShaderBuffers();
    try createSyncObjects();
    defer app.init.gpa.free(app.render_semaphores);
    try createCommandPool();
    try loadTextures();
    try loadShaders();
    defer c.slang_wrapper_destroy(app.slang_wrapper);

    log.info("beginning loop", .{});

    try wio.run(loop);
}

fn loop() !bool {
    // sync
    try chk(try app.device.waitForFences(
        (&app.fences[app.frame_index])[0..1],
        .true,
        std.math.maxInt(u64),
    ));
    try app.device.resetFences((&app.fences[app.frame_index])[0..1]);
    const next_image_res = try app.device.acquireNextImageKHR(
        app.swapchain,
        std.math.maxInt(u64),
        app.present_semaphores[app.frame_index],
        .null_handle,
    );
    try chk_swapchain(next_image_res.result);
    app.image_index = next_image_res.image_index;

    // update shader data
    var aspect: f32 = @floatFromInt(app.size.width);
    aspect /= @floatFromInt(app.size.height);
    app.shader_data.projection = za.perspective(45, aspect, 0.1, 32);

    // Convert from OpenGL depth [-1, 1] to Vulkan [0, 1]
    app.shader_data.projection.data[2][2] = app.shader_data.projection.data[2][2] * 0.5 + app.shader_data.projection.data[2][3] * 0.5;
    app.shader_data.projection.data[3][2] = app.shader_data.projection.data[3][2] * 0.5 + app.shader_data.projection.data[3][3] * 0.5;

    app.shader_data.view = za.Mat4.identity().translate(app.cam_pos);
    for (0..3) |i| {
        var x: f32 = @floatFromInt(i);
        x = (x - 1) * 3;
        const instance_pos = za.Vec3.new(x, 0, 0);
        app.shader_data.model[i] = za.Mat4.identity()
            .translate(instance_pos)
            .mul(za.Quat.fromEulerAngles(app.object_rotations[i]).toMat4());
    }
    const shader_data_buffer: [*]ShaderData = @ptrCast(@alignCast(
        app.shader_data_buffers[app.frame_index].allocation_info.pMappedData.?,
    ));
    @memcpy(
        shader_data_buffer,
        (&app.shader_data)[0..1],
    );

    // build command buffer
    const cmd_buf = app.command_buffers[app.frame_index];
    try app.device.resetCommandBuffer(cmd_buf, .{});
    try app.device.beginCommandBuffer(cmd_buf, &vk.CommandBufferBeginInfo{
        .flags = .{ .one_time_submit_bit = true },
    });

    app.device.cmdPipelineBarrier2(cmd_buf, &vk.DependencyInfo{
        .image_memory_barrier_count = 2,
        .p_image_memory_barriers = &[2]vk.ImageMemoryBarrier2{
            vk.ImageMemoryBarrier2{
                .src_stage_mask = .{ .color_attachment_output_bit = true },
                .src_access_mask = .{},
                .dst_stage_mask = .{ .color_attachment_output_bit = true },
                .dst_access_mask = .{
                    .color_attachment_read_bit = true,
                    .color_attachment_write_bit = true,
                },
                .old_layout = .undefined,
                .new_layout = .attachment_optimal,
                .image = app.swapchain_images[app.image_index],
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .level_count = 1,
                    .layer_count = 1,
                    .base_array_layer = 0,
                    .base_mip_level = 0,
                },
                .src_queue_family_index = 0,
                .dst_queue_family_index = 0,
            },
            vk.ImageMemoryBarrier2{
                .src_stage_mask = .{ .late_fragment_tests_bit = true },
                .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
                .dst_stage_mask = .{ .early_fragment_tests_bit = true },
                .dst_access_mask = .{ .depth_stencil_attachment_write_bit = true },
                .old_layout = .undefined,
                .new_layout = .attachment_optimal,
                .image = app.depth_image,
                .subresource_range = .{
                    .aspect_mask = .{ .depth_bit = true, .stencil_bit = true },
                    .level_count = 1,
                    .layer_count = 1,
                    .base_array_layer = 0,
                    .base_mip_level = 0,
                },
                .src_queue_family_index = 0,
                .dst_queue_family_index = 0,
            },
        },
    });

    app.device.cmdBeginRendering(
        cmd_buf,
        &std.mem.zeroInit(
            vk.RenderingInfo,
            .{
                .render_area = .{
                    .extent = .{
                        .width = @as(u32, app.size.width),
                        .height = @as(u32, app.size.height),
                    },
                    .offset = .{ .x = 0, .y = 0 },
                },
                .layer_count = 1,
                .color_attachment_count = 1,
                .p_color_attachments = @as(
                    [*]const vk.RenderingAttachmentInfo,
                    @ptrCast(&std.mem.zeroInit(
                        vk.RenderingAttachmentInfo,
                        .{
                            .image_view = app.swapchain_image_views[app.image_index],
                            .image_layout = .attachment_optimal,
                            .load_op = .clear,
                            .store_op = .store,
                            .clear_value = vk.ClearValue{
                                .color = .{
                                    .float_32 = .{ 0, 0, 0, 1 },
                                },
                            },
                        },
                    )),
                ),
                .p_depth_attachment = &std.mem.zeroInit(
                    vk.RenderingAttachmentInfo,
                    .{
                        .image_view = app.depth_image_view,
                        .image_layout = .attachment_optimal,
                        .load_op = .clear,
                        .store_op = .dont_care,
                        .clear_value = vk.ClearValue{
                            .depth_stencil = .{
                                .depth = 1,
                                .stencil = 0,
                            },
                        },
                    },
                ),
            },
        ),
    );

    app.device.cmdSetViewport(
        cmd_buf,
        0,
        (&vk.Viewport{
            .width = @floatFromInt(app.size.width),
            .height = @floatFromInt(app.size.height),
            .min_depth = 0,
            .max_depth = 1,
            .x = 0,
            .y = 0,
        })[0..1],
    );

    app.device.cmdBindPipeline(cmd_buf, .graphics, app.pipeline);

    app.device.cmdSetScissor(
        cmd_buf,
        0,
        (&vk.Rect2D{
            .extent = .{
                .width = @intCast(app.size.width),
                .height = @intCast(app.size.height),
            },
            .offset = .{ .x = 0, .y = 0 },
        })[0..1],
    );

    app.device.cmdBindDescriptorSets(
        cmd_buf,
        .graphics,
        app.pipeline_layout,
        0,
        (&app.descriptor_set_tex)[0..1],
        null,
    );

    app.device.cmdBindVertexBuffers(
        cmd_buf,
        0,
        (&app.vertex_buffer)[0..1],
        (&@as(u64, 0))[0..1],
    );

    const vertex_buffer_size = @sizeOf(Vertex) * app.vertices.len;
    app.device.cmdBindIndexBuffer(
        cmd_buf,
        app.vertex_buffer,
        vertex_buffer_size,
        .uint16,
    );

    app.device.cmdPushConstants(
        cmd_buf,
        app.pipeline_layout,
        .{ .vertex_bit = true },
        0,
        @sizeOf(vk.DeviceAddress),
        @ptrCast(&app.shader_data_buffers[app.frame_index].device_address),
    );

    app.device.cmdDrawIndexed(cmd_buf, @intCast(app.indices.len), 3, 0, 0, 0);
    app.device.cmdEndRendering(cmd_buf);

    app.device.cmdPipelineBarrier2(cmd_buf, &vk.DependencyInfo{
        .image_memory_barrier_count = 1,
        .p_image_memory_barriers = @ptrCast(&std.mem.zeroInit(
            vk.ImageMemoryBarrier2,
            .{
                .src_stage_mask = .{ .color_attachment_output_bit = true },
                .src_access_mask = .{ .color_attachment_write_bit = true },
                .dst_stage_mask = .{ .color_attachment_output_bit = true },
                .dst_access_mask = .{},
                .old_layout = .attachment_optimal,
                .new_layout = .present_src_khr,
                .image = app.swapchain_images[app.image_index],
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .level_count = 1,
                    .layer_count = 1,
                    .base_array_layer = 0,
                    .base_mip_level = 0,
                },
            },
        )),
    });

    try app.device.endCommandBuffer(cmd_buf);

    // submit to graphics queue
    try app.device.queueSubmit(
        app.queue,
        (&vk.SubmitInfo{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&app.present_semaphores[app.frame_index]),
            .p_wait_dst_stage_mask = @ptrCast(&vk.PipelineStageFlags{
                .color_attachment_output_bit = true,
            }),
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cmd_buf),
            .signal_semaphore_count = 1,
            .p_signal_semaphores = @ptrCast(&app.render_semaphores[app.image_index]),
        })[0..1],
        app.fences[app.frame_index],
    );

    app.frame_index = (app.frame_index + 1) % max_frames_in_flight;

    try chk_swapchain(try app.device.queuePresentKHR(
        app.queue,
        &vk.PresentInfoKHR{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&app.render_semaphores[app.image_index]),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast(&app.swapchain),
            .p_image_indices = @ptrCast(&app.image_index),
        },
    ));

    while (app.window.getEvent()) |event| {
        switch (event) {
            .close => return false,
            .size_physical => |new_size| {
                app.size = new_size;
                app.update_swapchain = true;
            },
            .button_press => |button| {
                switch (button) {
                    .left, .a => {
                        app.object_rotations[app.shader_data.selected].yMut().* -= app.rot_speed;
                    },
                    .right, .d => {
                        app.object_rotations[app.shader_data.selected].yMut().* += app.rot_speed;
                    },
                    .up, .w => {
                        app.object_rotations[app.shader_data.selected].xMut().* -= app.rot_speed;
                    },
                    .down, .s => {
                        app.object_rotations[app.shader_data.selected].xMut().* += app.rot_speed;
                    },
                    .space => {
                        app.shader_data.selected = (app.shader_data.selected + 1) % 3;
                    },
                    else => {},
                }
            },
            else => {},
        }
    }

    // resize swapchain
    if (app.update_swapchain) {
        app.update_swapchain = false;
        try app.device.deviceWaitIdle();
        app.surface_capabilities = try app.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(
            app.physical_device,
            app.surface,
        );
        const old_swapchain = app.swapchain;
        app.swapchain = try app.device.createSwapchainKHR(&vk.SwapchainCreateInfoKHR{
            .surface = app.surface,
            .old_swapchain = old_swapchain,
            .min_image_count = app.surface_capabilities.min_image_count,
            .image_format = app.image_format,
            .image_color_space = .srgb_nonlinear_khr,
            .image_extent = .{ .width = app.size.width, .height = app.size.height },
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true },
            .image_sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = &.{app.queue_family_index},
            .pre_transform = .{ .identity_bit_khr = true },
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = .fifo_khr,
            .clipped = .true,
        }, null);

        for (app.swapchain_image_views) |image_view| {
            app.device.destroyImageView(image_view, null);
        }

        var image_count: u32 = undefined;
        try chk(try app.device.getSwapchainImagesKHR(
            app.swapchain,
            &image_count,
            null,
        ));
        assert(image_count == app.swapchain_images.len);
        try chk(try app.device.getSwapchainImagesKHR(
            app.swapchain,
            &image_count,
            app.swapchain_images.ptr,
        ));

        for (0..image_count) |i| {
            app.swapchain_image_views[i] = try app.device.createImageView(
                &std.mem.zeroInit(vk.ImageViewCreateInfo, .{
                    .image = app.swapchain_images[i],
                    .view_type = .@"2d",
                    .format = app.image_format,
                    .subresource_range = .{
                        .aspect_mask = .{ .color_bit = true },
                        .level_count = 1,
                        .layer_count = 1,
                        .base_array_layer = 0,
                        .base_mip_level = 0,
                    },
                }),
                null,
            );
        }

        assert(image_count == app.render_semaphores.len);
        for (app.render_semaphores) |*semaphore| {
            app.device.destroySemaphore(semaphore.*, null);
            semaphore.* = try app.device.createSemaphore(&vk.SemaphoreCreateInfo{}, null);
        }

        app.device.destroySwapchainKHR(old_swapchain, null);
        c.vmaDestroyImage(
            app.vma_allocator,
            @ptrFromInt(@intFromEnum(app.depth_image)),
            app.depth_image_allocation,
        );
        app.device.destroyImageView(app.depth_image_view, null);

        try chk_c(c.vmaCreateImage(
            app.vma_allocator,
            @ptrCast(&vk.ImageCreateInfo{
                .image_type = .@"2d",
                .format = app.depth_format,
                .extent = .{
                    .width = app.size.width,
                    .height = app.size.height,
                    .depth = 1,
                },
                .mip_levels = 1,
                .array_layers = 1,
                .samples = .{ .@"1_bit" = true },
                .tiling = .optimal,
                .usage = .{ .depth_stencil_attachment_bit = true },
                .initial_layout = .undefined,
                .sharing_mode = .exclusive,
            }),
            &c.VmaAllocationCreateInfo{
                .flags = c.VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                .usage = c.VMA_MEMORY_USAGE_AUTO,
            },
            @ptrCast(&app.depth_image),
            &app.depth_image_allocation,
            null,
        ));

        app.depth_image_view = try app.device.createImageView(
            &std.mem.zeroInit(vk.ImageViewCreateInfo, .{
                .image = app.depth_image,
                .view_type = .@"2d",
                .format = app.depth_format,
                .subresource_range = .{
                    .aspect_mask = .{ .depth_bit = true },
                    .level_count = 1,
                    .layer_count = 1,
                    .base_array_layer = 0,
                    .base_mip_level = 0,
                },
            }),
            null,
        );
    }

    return true;
}

fn renderSplash(fb: *wio.Framebuffer, size: wio.Size, t: u16) void {
    var y: u32 = 0;
    while (y < size.height) : (y += 1) {
        var x: u32 = 0;
        while (x < size.width) : (x += 1) {
            const v = x ^ y ^ t;
            fb.setPixel(
                x,
                y,
                ((v & 0xFF) << 16) | (((v >> 1) & 0xFF) << 8) | ((v >> 2) & 0xFF),
            );
        }
    }
}

fn createInstance() !void {
    const arena = app.init.arena;
    defer _ = arena.reset(.retain_capacity);

    // required layers
    const required_layers = [_][]const u8{"VK_LAYER_KHRONOS_validation"};

    //  enabled layers
    const available_layers = try app.vk_base.enumerateInstanceLayerPropertiesAlloc(arena.allocator());
    for (available_layers) |layer| {
        log.debug("available instance layer:\t{s}\t{s}", .{ layer.layer_name, layer.description });
    }

    // check that all required layers are available
    for (required_layers) |required_layer_name| {
        var found = false;
        for (available_layers) |available_layer| {
            // std.debug.print("required instance layer: {s}\n", .{required_layer_name});
            // std.debug.print("available instance layer: {s}\n", .{available_layer.layer_name});
            const name = std.mem.sliceTo(&available_layer.layer_name, 0);
            if (std.mem.eql(u8, name, required_layer_name)) {
                log.info("required instance layer found: {s}", .{required_layer_name});
                found = true;
                break;
            }
        }
        if (!found) {
            log.err("required instance layer not found: {s}", .{required_layer_name});
            return error.instance_layer_not_found;
        }
    }

    const available_extensions = try app.vk_base.enumerateInstanceExtensionPropertiesAlloc(null, arena.allocator());
    for (available_extensions) |extension| {
        log.debug("available instance extension:\t{s}", .{extension.extension_name});
    }

    // required extensions
    const wio_required_extensions = wio.getRequiredVulkanInstanceExtensions();
    var required_extensions: std.ArrayList([*:0]const u8) = .empty;
    try required_extensions.ensureTotalCapacity(arena.allocator(), wio_required_extensions.len + 1);
    required_extensions.appendSliceBounded(wio_required_extensions) catch unreachable;
    required_extensions.appendBounded("VK_KHR_portability_enumeration") catch unreachable;

    // check that all required layers are available
    for (required_extensions.items) |required_extension| {
        var found = false;
        const required_name = std.mem.sliceTo(required_extension, 0);
        for (available_extensions) |available_extension| {
            const available_name = std.mem.sliceTo(&available_extension.extension_name, 0);
            if (std.mem.eql(u8, available_name, required_name)) {
                log.info("required instance extension found: {s}", .{required_name});
                found = true;
                break;
            }
        }
        if (!found) {
            log.err("required instance extension not found: {s}", .{required_name});
            return error.instance_extension_not_found;
        }
    }

    const handle = try app.vk_base.createInstance(&vk.InstanceCreateInfo{
        .flags = .{ .enumerate_portability_bit_khr = true },
        .p_application_info = &vk.ApplicationInfo{
            .application_version = 0,
            .engine_version = 0,
            .api_version = @bitCast(vk.API_VERSION_1_3),
        },
        .enabled_layer_count = @intCast(required_layers.len),
        .pp_enabled_layer_names = @ptrCast(&required_layers),
        .enabled_extension_count = @intCast(required_extensions.items.len),
        .pp_enabled_extension_names = required_extensions.items.ptr,
    }, null);

    app.vk_instance_handle = handle;
    app.vk_instance_wrapper = .load(handle, app.vk_base.dispatch.vkGetInstanceProcAddr.?);
    app.instance = .init(handle, &app.vk_instance_wrapper);
}

fn selectPhysicalDevice() !void {
    var device_count: u32 = 0;
    try chk(try app.instance.enumeratePhysicalDevices(&device_count, null));

    const arena = app.init.arena.allocator();
    defer _ = app.init.arena.reset(.retain_capacity);

    const devices = try arena.alloc(vk.PhysicalDevice, device_count);
    try chk(try app.instance.enumeratePhysicalDevices(&device_count, devices.ptr));

    var device_index: u32 = 0;
    const args = try app.init.minimal.args.toSlice(arena);
    for (args, 0..) |arg, i| {
        log.debug("args[{d}]: {s}", .{ i, arg });
    }
    if (args.len > 1) {
        device_index = try std.fmt.parseInt(u32, args[1], 10);
        std.debug.assert(device_index < device_count);
    }

    var device_properties = vk.PhysicalDeviceProperties2{ .properties = undefined };
    for (devices) |device| {
        app.instance.getPhysicalDeviceProperties2(device, &device_properties);
        log.debug("found device: {s} {any}", .{ device_properties.properties.device_name, device_properties.properties.device_type });
    }

    app.instance.getPhysicalDeviceProperties2(devices[device_index], &device_properties);

    log.info("selected device: {s}", .{device_properties.properties.device_name});
    app.physical_device = devices[device_index];
}

fn createSurface() !void {
    const result: vk.Result = @enumFromInt(
        app.window.vkCreateSurface(
            @intFromEnum(app.instance.handle),
            null,
            @ptrCast(&app.surface),
        ),
    );
    if (result != .success) {
        log.err("error during surface creation: {any}", .{result});
        return error.create_surface_failed;
    }
}

fn selectQueueFamily() !void {
    var queue_family_count: u32 = 0;
    app.instance.getPhysicalDeviceQueueFamilyProperties(app.physical_device, &queue_family_count, null);

    const arena = app.init.arena.allocator();
    defer _ = app.init.arena.reset(.retain_capacity);

    const queue_families = try arena.alloc(vk.QueueFamilyProperties, queue_family_count);
    app.instance.getPhysicalDeviceQueueFamilyProperties(app.physical_device, &queue_family_count, queue_families.ptr);

    var queue_family_index: u32 = 0;
    for (queue_families, 0..) |candidate, i| {
        const can_present = try app.instance.getPhysicalDeviceSurfaceSupportKHR(app.physical_device, queue_family_index, app.surface) == .true;
        if (candidate.queue_flags.graphics_bit and can_present) {
            queue_family_index = @intCast(i);
            break;
        }
    }

    log.info("selected graphics queue family: {d}", .{queue_family_index});
    app.queue_family_index = queue_family_index;
}

fn createLogicalDevice() !void {
    const qfpriorities = [_]f32{1.0};
    var queue_create_info = [_]vk.DeviceQueueCreateInfo{
        .{
            .queue_family_index = app.queue_family_index,
            .queue_count = 1,
            .p_queue_priorities = &qfpriorities,
        },
    };

    var enabled_vk12_features = vk.PhysicalDeviceVulkan12Features{
        .descriptor_indexing = .true,
        .shader_sampled_image_array_non_uniform_indexing = .true,
        .descriptor_binding_variable_descriptor_count = .true,
        .runtime_descriptor_array = .true,
        .buffer_device_address = .true,
    };

    const enabled_vk13_features = vk.PhysicalDeviceVulkan13Features{
        .p_next = &enabled_vk12_features,
        .synchronization_2 = .true,
        .dynamic_rendering = .true,
    };

    const enabled_vk10_features = vk.PhysicalDeviceFeatures{
        .sampler_anisotropy = .true,
    };

    const device_extensions = [_][*:0]const u8{"VK_KHR_swapchain"};

    const device_create_info = vk.DeviceCreateInfo{
        .p_next = &enabled_vk13_features,
        .queue_create_info_count = 1,
        .p_queue_create_infos = &queue_create_info,
        .enabled_extension_count = @intCast(device_extensions.len),
        .pp_enabled_extension_names = &device_extensions,
        .p_enabled_features = &enabled_vk10_features,
    };

    const logical_device_handle = try app.instance.createDevice(
        app.physical_device,
        &device_create_info,
        null,
    );

    app.vk_device_handle = logical_device_handle;

    app.vk_device_wrapper = .load(
        logical_device_handle,
        app.vk_instance_wrapper.dispatch.vkGetDeviceProcAddr.?,
    );
    app.device = .init(logical_device_handle, &app.vk_device_wrapper);
    log.info("created logical device", .{});

    app.queue = app.device.getDeviceQueue(app.queue_family_index, 0);
    log.info("created device queue", .{});
}

fn setupVMA() !void {
    const vk_functions: c.VmaVulkanFunctions = .{
        .vkGetInstanceProcAddr = @ptrCast(app.vk_base.dispatch.vkGetInstanceProcAddr),
        .vkGetDeviceProcAddr = @ptrCast(app.vk_instance_wrapper.dispatch.vkGetDeviceProcAddr),
        .vkCreateImage = @ptrCast(app.vk_device_wrapper.dispatch.vkCreateImage),
    };

    const allocator_ci: c.VmaAllocatorCreateInfo = .{
        .flags = c.VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = @ptrFromInt(@intFromEnum(app.physical_device)),
        .device = @ptrFromInt(@intFromEnum(app.vk_device_handle)),
        .pVulkanFunctions = &vk_functions,
        .instance = @ptrFromInt(@intFromEnum(app.vk_instance_handle)),
    };

    const res = c.vmaCreateAllocator(&allocator_ci, &app.vma_allocator);
    try chk_c(res);
}

fn getSurfaceCapabilities() !void {
    app.surface_capabilities = try app.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(app.physical_device, app.surface);
}

fn createSwapchain() !void {
    app.size.width = @truncate(std.math.clamp(
        app.size.width,
        app.surface_capabilities.min_image_extent.width,
        app.surface_capabilities.max_image_extent.width,
    ));
    app.size.height = @truncate(std.math.clamp(
        app.size.height,
        app.surface_capabilities.min_image_extent.height,
        app.surface_capabilities.max_image_extent.height,
    ));

    app.image_format = .b8g8r8a8_srgb;
    const image_format = app.image_format;
    app.swapchain = try app.device.createSwapchainKHR(&vk.SwapchainCreateInfoKHR{
        .surface = app.surface,
        .min_image_count = app.surface_capabilities.min_image_count,
        .image_format = image_format,
        .image_color_space = .srgb_nonlinear_khr,
        .image_extent = .{ .width = app.size.width, .height = app.size.height },
        .image_array_layers = 1,
        .image_usage = .{ .color_attachment_bit = true },
        .image_sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = &.{app.queue_family_index},
        .pre_transform = .{ .identity_bit_khr = true },
        .composite_alpha = .{ .opaque_bit_khr = true },
        .present_mode = .fifo_khr,
        .clipped = .true,
    }, null);
    log.info("swapchain created", .{});

    app.swapchain_images = try app.device.getSwapchainImagesAllocKHR(app.swapchain, app.init.gpa);

    app.swapchain_image_views = try app.init.gpa.alloc(vk.ImageView, app.swapchain_images.len);
    for (app.swapchain_images, app.swapchain_image_views) |image, *view| {
        view.* = try app.device.createImageView(&vk.ImageViewCreateInfo{
            .image = image,
            .view_type = .@"2d",
            .format = image_format,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
    }

    log.info("allocated {d} swapchain image views", .{app.swapchain_image_views.len});
}

fn recreateSwapchain() !void {
    try app.device.deviceWaitIdle();
}

fn createDepthAttachment() !void {
    log.debug("creating depth attachment (start)", .{});

    const depth_format_list = [_]vk.Format{ .d32_sfloat_s8_uint, .d24_unorm_s8_uint };
    var depth_format = vk.Format.undefined;
    for (depth_format_list) |format| {
        var format_properties: vk.FormatProperties2 = .{ .format_properties = .{} };
        app.instance.getPhysicalDeviceFormatProperties2(
            app.physical_device,
            format,
            &format_properties,
        );
        if (format_properties.format_properties.optimal_tiling_features.depth_stencil_attachment_bit) {
            depth_format = format;
            break;
        }
    }
    assert(depth_format != .undefined);

    app.depth_format = depth_format;
    log.debug("depth format selected: {any}", .{depth_format});

    try chk_c(c.vmaCreateImage(
        app.vma_allocator,
        @ptrCast(&vk.ImageCreateInfo{
            .image_type = .@"2d",
            .format = depth_format,
            .extent = .{
                .width = app.size.width,
                .height = app.size.height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{ .depth_stencil_attachment_bit = true },
            .initial_layout = .undefined,
            .sharing_mode = .exclusive,
        }),
        &c.VmaAllocationCreateInfo{
            .flags = c.VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            .usage = c.VMA_MEMORY_USAGE_AUTO,
        },
        @ptrCast(&app.depth_image),
        &app.depth_image_allocation,
        null,
    ));

    log.debug("depth attachment image created", .{});

    app.depth_image_view = try app.device.createImageView(
        &vk.ImageViewCreateInfo{
            .image = app.depth_image,
            .view_type = .@"2d",
            .format = depth_format,
            .subresource_range = .{
                .aspect_mask = .{ .depth_bit = true },
                .level_count = 1,
                .layer_count = 1,
                .base_mip_level = 0,
                .base_array_layer = 0,
            },
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
        },
        null,
    );

    log.info("created depth attachment", .{});
}

fn loadMesh() !void {
    app.model_obj = try obj.parseObj(app.init.gpa, @embedFile("assets/suzanne.obj"));
    log.info("loaded {d} meshes", .{app.model_obj.meshes.len});
    assert(app.model_obj.meshes.len > 0);
    log.info("\t{d} indices", .{app.model_obj.meshes[0].indices.len});
    log.info("\t{d} materials", .{app.model_obj.meshes[0].materials.len});
    log.info("\t{d} faces", .{app.model_obj.meshes[0].num_vertices.len});

    var total_vertices: usize = 0;
    for (app.model_obj.meshes[0].num_vertices) |count| {
        total_vertices += (count - 2) * 3;
    }

    app.vertices = try app.init.gpa.alloc(Vertex, total_vertices);
    app.indices = try app.init.gpa.alloc(u16, total_vertices);

    var v_i: usize = 0;
    var offset: usize = 0;
    for (app.model_obj.meshes[0].num_vertices) |face_v_count| {
        const face_indices = app.model_obj.meshes[0].indices[offset .. offset + face_v_count];

        // Triangulate face (Triangle Fan)
        for (1..face_v_count - 1) |i| {
            const local_idxs = [_]usize{ 0, i, i + 1 };
            for (local_idxs) |local_idx| {
                const index = face_indices[local_idx];
                const v: Vertex = .{
                    .pos = za.Vec3.new(
                        app.model_obj.vertices[index.vertex.? * 3],
                        -app.model_obj.vertices[index.vertex.? * 3 + 1],
                        app.model_obj.vertices[index.vertex.? * 3 + 2],
                    ),
                    .normal = za.Vec3.new(
                        app.model_obj.normals[index.normal.? * 3],
                        -app.model_obj.normals[index.normal.? * 3 + 1],
                        app.model_obj.normals[index.normal.? * 3 + 2],
                    ),
                    .uv = za.Vec2.new(
                        app.model_obj.tex_coords[index.tex_coord.? * 2],
                        1.0 - app.model_obj.tex_coords[index.tex_coord.? * 2 + 1],
                    ),
                };
                app.vertices[v_i] = v;
                app.indices[v_i] = @intCast(v_i);
                v_i += 1;
            }
        }
        offset += face_v_count;
    }

    log.debug("translated model", .{});

    const vertex_buf_size: vk.DeviceSize = @sizeOf(Vertex) * total_vertices;
    const index_buf_size: vk.DeviceSize = @sizeOf(u16) * total_vertices;
    var vertex_buf_alloc_info: c.VmaAllocationInfo = .{};
    try chk_c(c.vmaCreateBuffer(
        app.vma_allocator,
        @ptrCast(&vk.BufferCreateInfo{
            .size = vertex_buf_size + index_buf_size,
            .usage = .{ .vertex_buffer_bit = true, .index_buffer_bit = true },
            .sharing_mode = .exclusive,
        }),
        &c.VmaAllocationCreateInfo{
            .flags = c.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                c.VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                c.VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = c.VMA_MEMORY_USAGE_AUTO,
        },
        @ptrCast(&app.vertex_buffer),
        &app.vertex_buffer_allocation,
        &vertex_buf_alloc_info,
    ));

    const vertex_buf_ptr: [*]Vertex = @ptrCast(@alignCast(vertex_buf_alloc_info.pMappedData.?));
    @memcpy(vertex_buf_ptr[0..app.vertices.len], app.vertices);
    log.debug("uploaded vertices", .{});

    const index_buf_ptr: [*]u16 = @ptrCast(&vertex_buf_ptr[app.vertices.len]);
    @memcpy(index_buf_ptr[0..app.indices.len], app.indices);
    log.debug("uploaded indices", .{});
}

fn createShaderBuffers() !void {
    log.info("creating shader buffers", .{});
    for (0..max_frames_in_flight) |i| {
        try chk_c(c.vmaCreateBuffer(
            app.vma_allocator,
            @ptrCast(&vk.BufferCreateInfo{
                .size = @intCast(@sizeOf(ShaderData)),
                .usage = .{ .shader_device_address_bit = true },
                .sharing_mode = .exclusive,
            }),
            &c.VmaAllocationCreateInfo{
                .flags = c.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    c.VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                    c.VMA_ALLOCATION_CREATE_MAPPED_BIT,
                .usage = c.VMA_MEMORY_USAGE_AUTO,
            },
            @ptrCast(&app.shader_data_buffers[i].buffer),
            &app.shader_data_buffers[i].allocation,
            &app.shader_data_buffers[i].allocation_info,
        ));
        app.shader_data_buffers[i].device_address =
            app.device.getBufferDeviceAddress(
                &vk.BufferDeviceAddressInfo{
                    .buffer = app.shader_data_buffers[i].buffer,
                },
            );
    }
}

fn createSyncObjects() !void {
    for (0..max_frames_in_flight) |i| {
        app.fences[i] = try app.device.createFence(
            &vk.FenceCreateInfo{ .flags = .{ .signaled_bit = true } },
            null,
        );
        app.present_semaphores[i] = try app.device.createSemaphore(&.{}, null);
    }

    app.render_semaphores = try app.init.gpa.alloc(vk.Semaphore, app.swapchain_images.len);
    for (app.render_semaphores) |*semaphore| {
        semaphore.* = try app.device.createSemaphore(&.{}, null);
    }
}

fn createCommandPool() !void {
    log.info("creating command pool", .{});
    app.command_pool = try app.device.createCommandPool(
        &vk.CommandPoolCreateInfo{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = app.queue_family_index,
        },
        null,
    );
    try app.device.allocateCommandBuffers(
        &vk.CommandBufferAllocateInfo{
            .command_pool = app.command_pool,
            .command_buffer_count = max_frames_in_flight,
            .level = .primary,
        },
        &app.command_buffers,
    );
}

fn loadTextures() !void {
    log.info("loading textures", .{});

    const n_textures = app.textures.len;
    for (0..n_textures) |i| {
        log.debug("loading texture: {d}", .{i});

        var ktx_texture: *c.ktxTexture = undefined;

        assert(c.ktxTexture_CreateFromMemory(
            @ptrCast(suzanne_ktx_bytes[i].ptr),
            @intCast(suzanne_ktx_bytes[i].len),
            c.KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
            @ptrCast(&ktx_texture),
        ) == c.KTX_SUCCESS);

        const format = c.ktxTexture_GetVkFormat(ktx_texture);

        try chk_c(c.vmaCreateImage(
            app.vma_allocator,
            @ptrCast(&vk.ImageCreateInfo{
                .image_type = .@"2d",
                .format = @enumFromInt(format),
                .extent = .{
                    .width = ktx_texture.baseWidth,
                    .height = ktx_texture.baseHeight,
                    .depth = 1,
                },
                .mip_levels = ktx_texture.numLevels,
                .array_layers = 1,
                .samples = .{ .@"1_bit" = true },
                .tiling = .optimal,
                .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
                .initial_layout = .undefined,
                .sharing_mode = .exclusive,
            }),
            &c.VmaAllocationCreateInfo{ .usage = c.VMA_MEMORY_USAGE_AUTO },
            @ptrCast(&app.textures[i].image),
            &app.textures[i].allocation,
            null,
        ));

        app.textures[i].view = try app.device.createImageView(
            &vk.ImageViewCreateInfo{
                .image = app.textures[i].image,
                .view_type = .@"2d",
                .format = @enumFromInt(format),
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .level_count = ktx_texture.numLevels,
                    .layer_count = 1,
                    .base_mip_level = 0,
                    .base_array_layer = 0,
                },
                .components = .{
                    .r = .identity,
                    .g = .identity,
                    .b = .identity,
                    .a = .identity,
                },
            },
            null,
        );

        // upload
        var img_src_buffer: vk.Buffer = .null_handle;
        var img_src_allocation: c.VmaAllocation = null;
        var img_src_alloc_info: c.VmaAllocationInfo = .{};
        try chk_c(c.vmaCreateBuffer(
            app.vma_allocator,
            @ptrCast(&vk.BufferCreateInfo{
                .size = @intCast(ktx_texture.dataSize),
                .usage = .{ .transfer_src_bit = true },
                .sharing_mode = .exclusive,
            }),
            &c.VmaAllocationCreateInfo{
                .flags = c.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    c.VMA_ALLOCATION_CREATE_MAPPED_BIT,
                .usage = c.VMA_MEMORY_USAGE_AUTO,
            },
            @ptrCast(&img_src_buffer),
            &img_src_allocation,
            &img_src_alloc_info,
        ));

        // copy image to src buffer
        var copy_src: []u8 = undefined;
        copy_src.ptr = @ptrCast(ktx_texture.pData);
        copy_src.len = ktx_texture.dataSize;
        const copy_dst: [*]u8 = @ptrCast(img_src_alloc_info.pMappedData.?);
        @memcpy(copy_dst, copy_src);

        // copy src buffer to device image
        const fence_one_time = try app.device.createFence(
            &vk.FenceCreateInfo{},
            null,
        );
        var cmd_buffer_one_time: vk.CommandBuffer = undefined;
        try app.device.allocateCommandBuffers(
            &vk.CommandBufferAllocateInfo{
                .command_pool = app.command_pool,
                .command_buffer_count = 1,
                .level = .primary,
            },
            @ptrCast(&cmd_buffer_one_time),
        );
        try app.device.beginCommandBuffer(
            cmd_buffer_one_time,
            &vk.CommandBufferBeginInfo{
                .flags = .{ .one_time_submit_bit = true },
            },
        );
        app.device.cmdPipelineBarrier2(
            cmd_buffer_one_time,
            &vk.DependencyInfo{
                .image_memory_barrier_count = 1,
                .p_image_memory_barriers = @ptrCast(&vk.ImageMemoryBarrier2{
                    .src_stage_mask = .{},
                    .src_access_mask = .{},
                    .dst_stage_mask = .{ .all_transfer_bit = true },
                    .dst_access_mask = .{ .transfer_write_bit = true },
                    .old_layout = .undefined,
                    .new_layout = .transfer_dst_optimal,
                    .image = app.textures[i].image,
                    .subresource_range = .{
                        .aspect_mask = .{ .color_bit = true },
                        .level_count = ktx_texture.numLevels,
                        .layer_count = 1,
                        .base_array_layer = 0,
                        .base_mip_level = 0,
                    },
                    .dst_queue_family_index = 0,
                    .src_queue_family_index = 0,
                }),
            },
        );
        const copy_regions = try app.init.arena.allocator().alloc(vk.BufferImageCopy, ktx_texture.numLevels);
        for (0..ktx_texture.numLevels) |j| {
            var mip_offset: c.ktx_size_t = 0;
            assert(c.ktxTexture_GetImageOffset(
                ktx_texture,
                @intCast(j),
                0,
                0,
                &mip_offset,
            ) == c.KTX_SUCCESS);
            copy_regions[j] = vk.BufferImageCopy{
                .buffer_offset = mip_offset,
                .image_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = @intCast(j),
                    .layer_count = 1,
                    .base_array_layer = 0,
                },
                .image_extent = .{
                    .width = ktx_texture.baseWidth >> @intCast(j),
                    .height = ktx_texture.baseHeight >> @intCast(j),
                    .depth = 1,
                },
                .buffer_row_length = 0,
                .buffer_image_height = 0,
                .image_offset = .{ .x = 0, .y = 0, .z = 0 },
            };
        }
        app.device.cmdCopyBufferToImage(
            cmd_buffer_one_time,
            img_src_buffer,
            app.textures[i].image,
            .transfer_dst_optimal,
            copy_regions,
        );
        app.device.cmdPipelineBarrier2(
            cmd_buffer_one_time,
            &vk.DependencyInfo{
                .image_memory_barrier_count = 1,
                .p_image_memory_barriers = @ptrCast(&vk.ImageMemoryBarrier2{
                    .src_stage_mask = .{ .all_transfer_bit = true },
                    .src_access_mask = .{ .transfer_write_bit = true },
                    .dst_stage_mask = .{ .fragment_shader_bit = true },
                    .dst_access_mask = .{ .shader_read_bit = true },
                    .old_layout = .transfer_dst_optimal,
                    .new_layout = .read_only_optimal,
                    .image = app.textures[i].image,
                    .subresource_range = .{
                        .aspect_mask = .{ .color_bit = true },
                        .level_count = ktx_texture.numLevels,
                        .layer_count = 1,
                        .base_array_layer = 0,
                        .base_mip_level = 0,
                    },
                    .dst_queue_family_index = 0,
                    .src_queue_family_index = 0,
                }),
            },
        );
        try app.device.endCommandBuffer(cmd_buffer_one_time);

        const one_time_submit_info = [_]vk.SubmitInfo{.{
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cmd_buffer_one_time),
        }};
        try app.device.queueSubmit(app.queue, &one_time_submit_info, fence_one_time);
        try chk(try app.device.waitForFences(&[_]vk.Fence{fence_one_time}, .true, std.math.maxInt(u64)));

        app.device.destroyFence(fence_one_time, null);
        c.vmaDestroyBuffer(
            app.vma_allocator,
            @ptrFromInt(@intFromEnum(img_src_buffer)),
            img_src_allocation,
        );

        // sampler
        app.textures[i].sampler = try app.device.createSampler(
            &vk.SamplerCreateInfo{
                .mag_filter = .linear,
                .min_filter = .linear,
                .mipmap_mode = .linear,

                .address_mode_u = @enumFromInt(0),
                .address_mode_v = @enumFromInt(0),
                .address_mode_w = @enumFromInt(0),
                .mip_lod_bias = 0,

                .anisotropy_enable = .true,
                .max_anisotropy = 8,

                .compare_enable = .false,
                .compare_op = @enumFromInt(0),

                .min_lod = 0,
                .max_lod = @floatFromInt(ktx_texture.numLevels),
                .border_color = @enumFromInt(0),
                .unnormalized_coordinates = .false,
            },
            null,
        );

        c.ktxTexture_Destroy(ktx_texture);
        app.texture_descriptors[i] = vk.DescriptorImageInfo{
            .sampler = app.textures[i].sampler,
            .image_view = app.textures[i].view,
            .image_layout = .read_only_optimal,
        };
    }

    // descriptor (indexing)
    app.descriptor_set_layout_tex = try app.device.createDescriptorSetLayout(
        &vk.DescriptorSetLayoutCreateInfo{
            .p_next = @ptrCast(&vk.DescriptorSetLayoutBindingFlagsCreateInfo{
                .binding_count = 1,
                .p_binding_flags = @ptrCast(&vk.DescriptorBindingFlags{
                    .variable_descriptor_count_bit = true,
                }),
            }),
            .binding_count = 1,
            .p_bindings = @ptrCast(&vk.DescriptorSetLayoutBinding{
                .descriptor_type = .combined_image_sampler,
                .descriptor_count = @intCast(app.textures.len),
                .stage_flags = .{ .fragment_bit = true },
                .binding = 0,
            }),
        },
        null,
    );
    app.descriptor_pool = try app.device.createDescriptorPool(
        &vk.DescriptorPoolCreateInfo{
            .max_sets = 1,
            .pool_size_count = 1,
            .p_pool_sizes = @ptrCast(&vk.DescriptorPoolSize{
                .type = .combined_image_sampler,
                .descriptor_count = @intCast(app.textures.len),
            }),
        },
        null,
    );
    const variable_desc_count: u32 = @intCast(app.textures.len);
    try app.device.allocateDescriptorSets(
        &vk.DescriptorSetAllocateInfo{
            .p_next = @ptrCast(&vk.DescriptorSetVariableDescriptorCountAllocateInfo{
                .descriptor_set_count = 1,
                .p_descriptor_counts = @ptrCast(&variable_desc_count),
            }),
            .descriptor_pool = app.descriptor_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&app.descriptor_set_layout_tex),
        },
        @ptrCast(&app.descriptor_set_tex),
    );

    app.device.updateDescriptorSets(&[1]vk.WriteDescriptorSet{
        vk.WriteDescriptorSet{
            .dst_set = app.descriptor_set_tex,
            .dst_binding = 0,
            .descriptor_count = @intCast(app.texture_descriptors.len),
            .descriptor_type = .combined_image_sampler,
            .p_image_info = &app.texture_descriptors,

            .dst_array_element = 0,
            .p_buffer_info = undefined,
            .p_texel_buffer_view = undefined,
        },
    }, null);
}

fn loadShaders() !void {
    app.slang_wrapper = c.slang_wrapper_create().?;
    const spirv_buffer: *c.SpirvBuffer = c.slang_shader_compile(
        app.slang_wrapper,
        "triangle",
        "src/assets/shader.slang",
        // "assets/shader.slang",
    );
    defer c.slang_shader_free(spirv_buffer);
    const shader_module = try app.device.createShaderModule(
        &vk.ShaderModuleCreateInfo{
            .code_size = spirv_buffer.code_size,
            .p_code = spirv_buffer.code_ptr,
        },
        null,
    );

    app.pipeline = try createPipeline(shader_module);
}

fn createPipeline(shader_module: vk.ShaderModule) !vk.Pipeline {
    log.info("creating pipeline for shader module: {any}", .{shader_module});
    app.pipeline_layout = try app.device.createPipelineLayout(
        &vk.PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&app.descriptor_set_layout_tex),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&vk.PushConstantRange{
                .stage_flags = .{ .vertex_bit = true },
                .size = @sizeOf(vk.DeviceAddress),
                .offset = 0,
            }),
        },
        null,
    );

    var pipeline: vk.Pipeline = .null_handle;
    var pipelines: []vk.Pipeline = undefined;
    pipelines.ptr = @ptrCast(&pipeline);
    pipelines.len = 1;

    var pipeline_create_info = vk.GraphicsPipelineCreateInfo{
        .p_next = @ptrCast(&vk.PipelineRenderingCreateInfo{
            .color_attachment_count = 1,
            .p_color_attachment_formats = @ptrCast(&app.image_format),
            .depth_attachment_format = app.depth_format,
            .stencil_attachment_format = .undefined,
            .view_mask = 0,
        }),
        .stage_count = 2,
        .p_stages = &[_]vk.PipelineShaderStageCreateInfo{
            vk.PipelineShaderStageCreateInfo{
                .stage = .{ .vertex_bit = true },
                .module = shader_module,
                .p_name = "main",
            },
            vk.PipelineShaderStageCreateInfo{
                .stage = .{ .fragment_bit = true },
                .module = shader_module,
                .p_name = "main",
            },
        },
        .p_vertex_input_state = &vk.PipelineVertexInputStateCreateInfo{
            .vertex_binding_description_count = 1,
            .p_vertex_binding_descriptions = @ptrCast(&vk.VertexInputBindingDescription{
                .binding = 0,
                .stride = @sizeOf(Vertex),
                .input_rate = .vertex,
            }),
            .vertex_attribute_description_count = 3,
            .p_vertex_attribute_descriptions = @ptrCast(&[_]vk.VertexInputAttributeDescription{
                vk.VertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .offset = @offsetOf(Vertex, "pos"),
                    .format = .r32g32b32_sfloat,
                },
                vk.VertexInputAttributeDescription{
                    .binding = 0,
                    .location = 1,
                    .offset = @offsetOf(Vertex, "normal"),
                    .format = .r32g32b32_sfloat,
                },
                vk.VertexInputAttributeDescription{
                    .binding = 0,
                    .location = 2,
                    .offset = @offsetOf(Vertex, "uv"),
                    .format = .r32g32_sfloat,
                },
            }),
        },
        .p_input_assembly_state = &vk.PipelineInputAssemblyStateCreateInfo{
            .topology = .triangle_list,
            .primitive_restart_enable = .false,
        },
        .p_viewport_state = &vk.PipelineViewportStateCreateInfo{
            .viewport_count = 1,
            .scissor_count = 1,
        },
        .p_rasterization_state = &std.mem.zeroInit(
            vk.PipelineRasterizationStateCreateInfo,
            .{
                .line_width = 1,
            },
        ),
        .p_multisample_state = &vk.PipelineMultisampleStateCreateInfo{
            .rasterization_samples = .{ .@"1_bit" = true },
            .sample_shading_enable = .false,
            .min_sample_shading = 0,
            .alpha_to_coverage_enable = .false,
            .alpha_to_one_enable = .false,
        },
        .p_depth_stencil_state = &std.mem.zeroInit(
            vk.PipelineDepthStencilStateCreateInfo,
            .{
                .depth_test_enable = .true,
                .depth_write_enable = .true,
                .depth_compare_op = .less_or_equal,
            },
        ),
        .p_color_blend_state = &vk.PipelineColorBlendStateCreateInfo{
            .attachment_count = 1,
            .p_attachments = @ptrCast(
                &std.mem.zeroInit(
                    vk.PipelineColorBlendAttachmentState,
                    .{
                        .color_write_mask = .{
                            .r_bit = true,
                            .g_bit = true,
                            .b_bit = true,
                            .a_bit = true,
                        },
                    },
                ),
            ),
            .logic_op = .clear,
            .logic_op_enable = .false,
            .blend_constants = [4]f32{ 0, 0, 0, 0 },
        }, // TODO:
        .p_dynamic_state = &vk.PipelineDynamicStateCreateInfo{
            .dynamic_state_count = 2,
            .p_dynamic_states = @ptrCast(&[2]vk.DynamicState{
                .viewport,
                .scissor,
            }),
        }, // TODO:
        .layout = app.pipeline_layout,

        .base_pipeline_index = 0,
        .subpass = 0,
    };

    var pipeline_create_infos: []vk.GraphicsPipelineCreateInfo = undefined;
    pipeline_create_infos.ptr = @ptrCast(&pipeline_create_info);
    pipeline_create_infos.len = 1;

    try chk(try app.device.createGraphicsPipelines(
        .null_handle,
        pipeline_create_infos,
        null,
        pipelines,
    ));

    assert(pipeline != .null_handle);
    return pipeline;
}

fn chk(result: vk.Result) !void {
    if (result != .success) {
        log.err("encountered vulkan error result: {any}", .{result});
        return error.vulkan_unknown_error;
    }
}

fn chk_c(result: c.VkResult) !void {
    try chk(@enumFromInt(result));
}

fn chk_swapchain(result: vk.Result) !void {
    if (result == .error_out_of_date_khr or result == .suboptimal_khr) {
        app.update_swapchain = true;
        return;
    }
    return chk(result);
}
