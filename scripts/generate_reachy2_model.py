#!/usr/bin/env python3
"""Generate the dependency-free Reachy2 C++ kinematics/sphere tables.

Usage:
  python3 scripts/generate_reachy2_model.py \
      /path/to/reachy2_spherized.urdf /path/to/reachy2.srdf \
      src/ompl/robots/Reachy2.h

Only the two seven-joint arms are active. Every other movable joint is fixed at
zero; gripper mimic joints therefore remain closed at their URDF zero pose.

The generated model defaults to Reachy's grounded mobile-base pose.  The URDF
root is at the centre of the mobile base, not at its contact plane; the visual
mesh's -0.1075 m origin records that offset.  Keeping the root pose explicit
also lets a future mobile-manipulation state supply x/y/yaw without regenerating
the arm model.
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


ACTIVE = (
    "l_shoulder_pitch", "l_shoulder_roll", "l_elbow_yaw", "l_elbow_pitch",
    "l_wrist_roll", "l_wrist_pitch", "l_wrist_yaw",
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
    "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw",
)


def vec(text, default="0 0 0"):
    return tuple(float(x) for x in (text or default).split())


def fmt(x):
    if abs(x) < 5e-16:
        x = 0.0
    return f"{x:.17g}"


def rotation_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (
        (cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
        (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
        (-sp, cp*sr, cp*cr),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("urdf", type=Path)
    ap.add_argument("srdf", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    robot = ET.parse(args.urdf).getroot()
    links = [x.get("name") for x in robot.findall("link")]
    children = {j.find("child").get("link") for j in robot.findall("joint")}
    roots = [x for x in links if x not in children]
    if len(roots) != 1:
        raise SystemExit(f"expected one root link, got {roots}")

    by_parent = {}
    for j in robot.findall("joint"):
        by_parent.setdefault(j.find("parent").get("link"), []).append(j)

    ordered_links = [roots[0]]
    records = []
    link_index = {roots[0]: 0}
    queue = [roots[0]]
    active_index = {name: i for i, name in enumerate(ACTIVE)}
    joint_limits = {}
    while queue:
        parent = queue.pop(0)
        for j in by_parent.get(parent, []):
            child = j.find("child").get("link")
            link_index[child] = len(ordered_links)
            ordered_links.append(child)
            queue.append(child)
            origin = j.find("origin")
            xyz = vec(origin.get("xyz") if origin is not None else None)
            rpy = vec(origin.get("rpy") if origin is not None else None)
            axis_node = j.find("axis")
            axis = vec(axis_node.get("xyz") if axis_node is not None else "1 0 0")
            ai = active_index.get(j.get("name"), -1)
            records.append((link_index[parent], child, ai, xyz, rotation_rpy(rpy), axis))
            if ai >= 0:
                lim = j.find("limit")
                joint_limits[j.get("name")] = (
                    float(lim.get("lower", -math.pi)), float(lim.get("upper", math.pi)),
                    float(lim.get("velocity", 1.0)))

    if any(name not in joint_limits for name in ACTIVE):
        raise SystemExit("one or more active joints are missing from the URDF")

    # Ancestor-active mask for each link.
    masks = [0] * len(ordered_links)
    for parent, child, ai, *_ in records:
        masks[link_index[child]] = masks[parent] | (0 if ai < 0 else 1 << ai)

    spheres = []
    for link in robot.findall("link"):
        li = link_index[link.get("name")]
        for collision in link.findall("collision"):
            sphere = collision.find("geometry/sphere")
            if sphere is None:
                continue
            origin = collision.find("origin")
            center = vec(origin.get("xyz") if origin is not None else None)
            spheres.append((li, center, float(sphere.get("radius")), link.get("name"), masks[li]))

    disabled = {
        tuple(sorted((x.get("link1"), x.get("link2"))))
        for x in ET.parse(args.srdf).getroot().findall("disable_collisions")
    }
    pairs = []
    for a in range(len(spheres)):
        for b in range(a + 1, len(spheres)):
            la, lb = spheres[a][3], spheres[b][3]
            if la == lb or tuple(sorted((la, lb))) in disabled:
                continue
            # A pair that no active arm joint can change is a static fact, not a
            # CBF row. Static robot/environment clearance belongs in the scene.
            if spheres[a][4] == spheres[b][4] == 0:
                continue
            pairs.append((a, b))

    def v3(v):
        return "{" + ", ".join(fmt(x) for x in v) + "}"
    def m3(m):
        return "{" + ", ".join(fmt(x) for row in m for x in row) + "}"

    lines = [
        "#pragma once", "", "// Generated by scripts/generate_reachy2_model.py; do not hand-edit.",
        "#include <array>", "#include <cstdint>", "#include <Eigen/Core>", "#include <Eigen/Geometry>", "",
        "namespace ompl::robots {", "class Reachy2 {", "public:",
        f"  static constexpr std::size_t nJoints = {len(ACTIVE)};",
        f"  static constexpr std::size_t nLinks = {len(ordered_links)};",
        f"  static constexpr std::size_t nSpheres = {len(spheres)};",
        f"  static constexpr std::size_t nSelfPairs = {len(pairs)};",
        "  using Configuration = Eigen::Matrix<double, nJoints, 1>;",
        "  using Jacobian = Eigen::Matrix<double, 3, nJoints>;",
        "  struct LinkStep { int parent; int active; std::array<double,3> xyz; std::array<double,9> rotation; std::array<double,3> axis; };",
        "  struct Sphere { int link; std::array<double,3> center; double radius; std::uint16_t influence; const char *name; };",
        "  struct SelfPair { std::uint16_t a, b; };",
        "  struct Kinematics { std::array<Eigen::Isometry3d,nLinks> link; std::array<Eigen::Vector3d,nJoints> origin, axis; };",
        "  explicit Reachy2(const Eigen::Isometry3d &basePose = groundedBasePose()) : basePose_(basePose) {}",
        "  static Eigen::Isometry3d groundedBasePose() { Eigen::Isometry3d pose=Eigen::Isometry3d::Identity(); pose.translation().z()=0.1075; return pose; }",
        "  const Eigen::Isometry3d &basePose() const { return basePose_; }",
        "  static const std::array<const char*,nJoints>& jointNames() { static const std::array<const char*,nJoints> v{{" + ",".join(f'\"{x}\"' for x in ACTIVE) + "}}; return v; }",
    ]
    for label, col in (("lowerBounds", 0), ("upperBounds", 1), ("velocityLimits", 2)):
        vals = ",".join(fmt(joint_limits[x][col]) for x in ACTIVE)
        lines.append(f"  static Configuration {label}() {{ Configuration q; q << {vals}; return q; }}")
    lines += [
        "  static const std::array<LinkStep,nLinks-1>& steps() { static const std::array<LinkStep,nLinks-1> v{{",
    ]
    for parent, child, ai, xyz, rot, axis in records:
        lines.append(f"    {{{parent},{ai},{v3(xyz)},{m3(rot)},{v3(axis)}}}, // {child}")
    lines += ["  }}; return v; }", "  static const std::array<Sphere,nSpheres>& spheres() { static const std::array<Sphere,nSpheres> v{{"]
    for li, center, radius, name, mask in spheres:
        lines.append(f"    {{{li},{v3(center)},{fmt(radius)},{mask},\"{name}\"}},")
    lines += ["  }}; return v; }", "  static const std::array<SelfPair,nSelfPairs>& selfPairs() { static const std::array<SelfPair,nSelfPairs> v{{"]
    lines += [f"    {{{a},{b}}}," for a, b in pairs]
    lines += [
        "  }}; return v; }",
        f"  static constexpr int leftTipLink = {link_index['l_arm_tip_bottom']};",
        f"  static constexpr int rightTipLink = {link_index['r_arm_tip_bottom']};",
        "  static Kinematics kinematicsAtBase(const Configuration &q, const Eigen::Isometry3d &basePose) {",
        "    Kinematics k; k.link[0] = basePose;",
        "    const auto &s=steps(); for(std::size_t i=0;i<s.size();++i){ const auto &x=s[i]; Eigen::Isometry3d o=Eigen::Isometry3d::Identity();",
        "      Eigen::Matrix3d R; for(int r=0;r<3;++r) for(int c=0;c<3;++c) R(r,c)=x.rotation[3*r+c]; o.linear()=R; o.translation()=Eigen::Vector3d(x.xyz.data());",
        "      Eigen::Isometry3d jf=k.link[x.parent]*o; if(x.active>=0){ Eigen::Vector3d a(x.axis.data()); k.origin[x.active]=jf.translation(); k.axis[x.active]=jf.linear()*a; jf.rotate(Eigen::AngleAxisd(q[x.active],a)); } k.link[i+1]=jf; } return k; }",
        "  Kinematics kinematics(const Configuration &q) const { return kinematicsAtBase(q, basePose_); }",
        "  static Eigen::Vector3d sphereCenter(const Kinematics&k,std::size_t i){ const auto&s=spheres()[i]; return k.link[s.link]*Eigen::Vector3d(s.center.data()); }",
        "  static Jacobian pointJacobian(const Kinematics&k,const Eigen::Vector3d&p,std::uint16_t mask){ Jacobian J=Jacobian::Zero(); for(int j=0;j<(int)nJoints;++j) if(mask&(1u<<j)) J.col(j)=k.axis[j].cross(p-k.origin[j]); return J; }",
        "  static Jacobian sphereJacobian(const Kinematics&k,std::size_t i){ const auto&s=spheres()[i]; auto p=sphereCenter(k,i); return pointJacobian(k,p,s.influence); }",
        "  static Eigen::Vector3d tipPosition(const Kinematics&k,bool left){ return k.link[left?leftTipLink:rightTipLink].translation(); }",
        "  static Jacobian tipJacobian(const Kinematics&k,bool left){ int li=left?leftTipLink:rightTipLink; auto p=k.link[li].translation(); return pointJacobian(k,p,left?" + str(masks[link_index['l_arm_tip_bottom']]) + ":" + str(masks[link_index['r_arm_tip_bottom']]) + "); }",
        "private:",
        "  Eigen::Isometry3d basePose_;",
        "};", "} // namespace ompl::robots", "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}: {len(ACTIVE)} joints, {len(spheres)} spheres, {len(pairs)} self pairs")


if __name__ == "__main__":
    main()
