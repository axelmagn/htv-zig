const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // TODO: error message on null
    const vulkan_sdk_path = b.graph.environ_map.get("VULKAN_SDK").?;
    std.debug.print("VULKAN_SDK:\t{s}\n", .{vulkan_sdk_path});

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    const vulkan_include_path = b.pathJoin(&[_][]const u8{ vulkan_sdk_path, "include" });
    exe_mod.addIncludePath(std.Build.LazyPath{ .cwd_relative = vulkan_include_path });
    // exe_mod.addIncludePath(b.path("external/tinyobj"));
    exe_mod.addIncludePath(b.path("src"));
    exe_mod.addCSourceFile(.{
        .file = b.path("src/vendor.cpp"),
        .flags = &[_][]const u8{"-std=c++20"},
        .language = .cpp,
    });

    const obj_mod = b.dependency("obj", .{ .target = target, .optimize = optimize }).module("obj");
    exe_mod.addImport("obj", obj_mod);

    const zalgebra_mod = b.dependency("zalgebra", .{ .target = target, .optimize = optimize }).module("zalgebra");
    exe_mod.addImport("zalgebra", zalgebra_mod);

    const mr_texture_mod = b.dependency("mr_texture", .{ .target = target, .optimize = optimize }).module("mr_texture");
    exe_mod.addImport("mr_texture", mr_texture_mod);

    const mr_ktx2_mod = b.dependency("mr_ktx2", .{ .target = target, .optimize = optimize }).module("mr_ktx2");
    exe_mod.addImport("mr_ktx2", mr_ktx2_mod);

    const wio = b.dependency("wio", .{
        .target = target,
        .optimize = optimize,
        .enable_vulkan = true,
        .enable_framebuffer = true,
        .unix_backends = b.option([]const u8, "unix_backends", "List of enabled wio backends"),
    });
    exe_mod.addImport("wio", wio.module("wio"));

    const vulkan_headers = b.dependency("vulkan_headers", .{});
    const vulkan = b.dependency("vulkan", .{ .registry = vulkan_headers.path("registry/vk.xml") });
    exe_mod.addImport("vulkan", vulkan.module("vulkan-zig"));

    const ktx_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ktx_mod.addIncludePath(b.path("external/ktx/include"));
    ktx_mod.addIncludePath(std.Build.LazyPath{ .cwd_relative = vulkan_include_path });
    ktx_mod.addCSourceFiles(.{
        .files = &[_][]const u8{
            "external/ktx/lib/texture.c",
            "external/ktx/lib/texture.c",
            "external/ktx/lib/hashlist.c",
            "external/ktx/lib/checkheader.c",
            "external/ktx/lib/swap.c",
            "external/ktx/lib/memstream.c",
            "external/ktx/lib/filestream.c",
            "external/ktx/lib/vkloader.c",
        },
        .language = .c,
    });

    const ktx_lib = b.addLibrary(.{
        .name = "ktx",
        .root_module = ktx_mod,
        .linkage = .static,
    });

    b.installArtifact(ktx_lib);

    exe_mod.linkLibrary(ktx_lib);
    exe_mod.addIncludePath(b.path("external/ktx/include"));

    const exe = b.addExecutable(.{
        .name = "vulkan",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
