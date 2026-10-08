// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file KarrasTreeTraverser.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include <vector>

namespace shamtree {

    /**
     * @struct KarrasTreeTraverser
     * @brief Utility struct to traverse a Karras Radix Tree
     */
    struct KarrasTreeTraverser;

    /// host version of the traverser
    struct KarrasTreeTraverserHost;

    /// read only accessor to buffer data
    struct KarrasTreeTraverserAccessed;

} // namespace shamtree

struct shamtree::KarrasTreeTraverserAccessed {
    const u32 *lchild_id;
    const u32 *rchild_id;
    const u8 *lchild_flag;
    const u8 *rchild_flag;
    u32 offset_leaf;

    /**
     * @brief Retrieves the left child node identifier for a given node ID.
     *
     * @param id The identifier of the node for which to find the left child.
     * @return The ID of the left child node, adjusted by the offset if the node is a leaf.
     */
    inline u32 get_left_child(u32 id) const {
        return lchild_id[id] + offset_leaf * u32(lchild_flag[id]);
    }

    /**
     * @brief Retrieves the right child node identifier for a given node ID.
     *
     * @param id The identifier of the node for which to find the right child.
     * @return The ID of the right child node, adjusted by the offset if the node is a leaf.
     */
    inline u32 get_right_child(u32 id) const {
        return rchild_id[id] + offset_leaf * u32(rchild_flag[id]);
    }

    /// is the given id a leaf (Note that if there is no internal cell every node is a leaf)
    inline bool is_id_leaf(u32 id) const { return id >= offset_leaf; }

    /// stack based tree traversal
    template<u32 tree_depth, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal(
        u32 root_node,
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        static constexpr u32 _nindex = 4294967295;

        // Init the stack state
        std::array<u32, tree_depth> id_stack;

        u32 stack_cursor       = tree_depth - 1;
        id_stack[stack_cursor] = root_node;

        // until the stack is empty
        while (stack_cursor < tree_depth) {

            // Pop the top of the stack
            u32 current_node_id    = id_stack[stack_cursor];
            id_stack[stack_cursor] = _nindex;
            stack_cursor++;

            // check iteraction creteria
            bool cur_id_valid = traverse_condition(current_node_id);

            if (cur_id_valid) { // leaf or cell satisfies the criteria

                if (is_id_leaf(current_node_id)) { // I found a leaf !!!!!

                    on_found_leaf(current_node_id);

                } else { // it can interact & not leaf => stack

                    u32 lid = get_left_child(current_node_id);
                    u32 rid = get_right_child(current_node_id);

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // This does not satisfy the criteria => excluded case (gravity for ex.)
                on_excluded_node(current_node_id);
            }
        }
    }

    /// stack based tree traversal
    template<u32 tree_depth, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal(
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        // On a Karras tree, the root is always 0
        u32 root_node = 0;

        stack_based_traversal<tree_depth>(
            root_node,
            std::forward<Functor1>(traverse_condition),
            std::forward<Functor2>(on_found_leaf),
            std::forward<Functor3>(on_excluded_node));
    }

    /**
     * @brief Persistent variant of the stack based tree traversal (for persistent kernels)
     *
     * Instead of exiting once the stack is empty, `next_work()` is called to update the caller's
     * internal state (e.g. store the previous result and fetch the next ray) and returns whether
     * a new traversal from `root_node` should start. `next_work()` is also called before the first
     * traversal, so the state does not need to be initialized beforehand.
     *
     * Everything runs in a single while loop, so that work items finishing at different times
     * within a warp do not leave threads idling in a nested loop.
     *
     * @param root_node the node from which every traversal starts
     * @param next_work `() -> bool`, update the internal state, return true to restart the
     * traversal and false to stop
     * @param traverse_condition `(u32 node_id) -> bool`, whether to traverse the node
     * @param on_found_leaf `(u32 node_id)`, called on each leaf satisfying the condition
     * @param on_excluded_node `(u32 node_id)`, called on each node not satisfying the condition
     */
    template<u32 tree_depth, class FunctorNext, class Functor1, class Functor2, class Functor3>
    inline void persistent_stack_based_traversal(
        u32 root_node,
        FunctorNext &&next_work,
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        // Init the stack state (empty, the root is pushed when fetching the first work item)
        std::array<u32, tree_depth> id_stack;

        u32 stack_cursor = tree_depth;

        // single loop over both the work items and the traversal
        while (true) {

            // the stack is empty => the current traversal is done, fetch the next work item
            if (stack_cursor >= tree_depth) {
                if (!next_work()) {
                    break;
                }

                stack_cursor           = tree_depth - 1;
                id_stack[stack_cursor] = root_node;
            }

            // Pop the top of the stack
            u32 current_node_id = id_stack[stack_cursor];
            stack_cursor++;

            // check iteraction creteria
            bool cur_id_valid = traverse_condition(current_node_id);

            if (cur_id_valid) { // leaf or cell satisfies the criteria

                if (is_id_leaf(current_node_id)) { // I found a leaf !!!!!

                    on_found_leaf(current_node_id);

                } else { // it can interact & not leaf => stack

                    u32 lid = get_left_child(current_node_id);
                    u32 rid = get_right_child(current_node_id);

                    id_stack[stack_cursor - 1] = rid;
                    stack_cursor--;

                    id_stack[stack_cursor - 1] = lid;
                    stack_cursor--;
                }
            } else {
                // This does not satisfy the criteria => excluded case (gravity for ex.)
                on_excluded_node(current_node_id);
            }
        }
    }

    /// Persistent variant of the stack based tree traversal (root = 0)
    template<u32 tree_depth, class FunctorNext, class Functor1, class Functor2, class Functor3>
    inline void persistent_stack_based_traversal(
        FunctorNext &&next_work,
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        // On a Karras tree, the root is always 0
        u32 root_node = 0;

        persistent_stack_based_traversal<tree_depth>(
            root_node,
            std::forward<FunctorNext>(next_work),
            std::forward<Functor1>(traverse_condition),
            std::forward<Functor2>(on_found_leaf),
            std::forward<Functor3>(on_excluded_node));
    }

    /**
     * @brief Stack based tree traversal with warp coherent leaf processing ("while-while")
     *
     * Same visiting order as stack_based_traversal, but the internal nodes are traversed in an
     * inner loop that only exits once a leaf satisfying the criteria is found (or the stack is
     * empty). The threads of a warp therefore reconverge before processing their leaves, instead
     * of processing them at different iterations of a single loop (see Aila & Laine 2009).
     */
    template<u32 tree_depth, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal_leaf_coherent(
        u32 root_node,
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        static constexpr u32 _nindex = 4294967295;

        // Init the stack state
        std::array<u32, tree_depth> id_stack;

        u32 stack_cursor       = tree_depth - 1;
        id_stack[stack_cursor] = root_node;

        // until the stack is empty
        while (stack_cursor < tree_depth) {

            u32 found_leaf = _nindex;

            // traverse the internal nodes until a leaf is found
            while (stack_cursor < tree_depth) {

                // Pop the top of the stack
                u32 current_node_id    = id_stack[stack_cursor];
                id_stack[stack_cursor] = _nindex;
                stack_cursor++;

                // check iteraction creteria
                bool cur_id_valid = traverse_condition(current_node_id);

                if (cur_id_valid) { // leaf or cell satisfies the criteria

                    if (is_id_leaf(current_node_id)) { // I found a leaf !!!!!

                        found_leaf = current_node_id;
                        break;

                    } else { // it can interact & not leaf => stack

                        u32 lid = get_left_child(current_node_id);
                        u32 rid = get_right_child(current_node_id);

                        id_stack[stack_cursor - 1] = rid;
                        stack_cursor--;

                        id_stack[stack_cursor - 1] = lid;
                        stack_cursor--;
                    }
                } else {
                    // This does not satisfy the criteria => excluded case (gravity for ex.)
                    on_excluded_node(current_node_id);
                }
            }

            if (found_leaf != _nindex) {
                on_found_leaf(found_leaf);
            }
        }
    }

    /// Warp coherent leaf processing variant of the stack based tree traversal (root = 0)
    template<u32 tree_depth, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal_leaf_coherent(
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        // On a Karras tree, the root is always 0
        u32 root_node = 0;

        stack_based_traversal_leaf_coherent<tree_depth>(
            root_node,
            std::forward<Functor1>(traverse_condition),
            std::forward<Functor2>(on_found_leaf),
            std::forward<Functor3>(on_excluded_node));
    }

    /// stack based tree traversal using a stack supplied by the caller instead of an
    /// internal std::array (e.g. a slice of a local_accessor for shared memory offload).
    /// `stack` is a functor `(u32 id) -> u32 &` giving access to the entry `id` of the stack.
    /// The stack must hold at least `stack_size` entries, which is enough for a traversal if
    /// `stack_size >= tree depth + 1`.
    template<class StackAccessor, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal(
        StackAccessor &&stack,
        u32 stack_size,
        u32 root_node,
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        static constexpr u32 _nindex = 4294967295;

        // Init the stack state
        u32 stack_cursor    = stack_size - 1;
        stack(stack_cursor) = root_node;

        // until the stack is empty
        while (stack_cursor < stack_size) {

            // Pop the top of the stack
            u32 current_node_id = stack(stack_cursor);
            stack(stack_cursor) = _nindex;
            stack_cursor++;

            // check iteraction creteria
            bool cur_id_valid = traverse_condition(current_node_id);

            if (cur_id_valid) { // leaf or cell satisfies the criteria

                if (is_id_leaf(current_node_id)) { // I found a leaf !!!!!

                    on_found_leaf(current_node_id);

                } else { // it can interact & not leaf => stack

                    u32 lid = get_left_child(current_node_id);
                    u32 rid = get_right_child(current_node_id);

                    stack(stack_cursor - 1) = rid;
                    stack_cursor--;

                    stack(stack_cursor - 1) = lid;
                    stack_cursor--;
                }
            } else {
                // This does not satisfy the criteria => excluded case (gravity for ex.)
                on_excluded_node(current_node_id);
            }
        }
    }

    /// stack based tree traversal using a stack supplied by the caller (root = 0)
    template<class StackAccessor, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal(
        StackAccessor &&stack,
        u32 stack_size,
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        // On a Karras tree, the root is always 0
        u32 root_node = 0;

        stack_based_traversal(
            std::forward<StackAccessor>(stack),
            stack_size,
            root_node,
            std::forward<Functor1>(traverse_condition),
            std::forward<Functor2>(on_found_leaf),
            std::forward<Functor3>(on_excluded_node));
    }

    /// Warp coherent leaf processing variant (see stack_based_traversal_leaf_coherent) using a
    /// stack supplied by the caller instead of an internal std::array.
    /// `stack` is a functor `(u32 id) -> u32 &` giving access to the entry `id` of the stack,
    /// which must hold at least `stack_size >= tree depth + 1` entries.
    template<class StackAccessor, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal_leaf_coherent(
        StackAccessor &&stack,
        u32 stack_size,
        u32 root_node,
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        static constexpr u32 _nindex = 4294967295;

        // Init the stack state
        u32 stack_cursor    = stack_size - 1;
        stack(stack_cursor) = root_node;

        // until the stack is empty
        while (stack_cursor < stack_size) {

            u32 found_leaf = _nindex;

            // traverse the internal nodes until a leaf is found
            while (stack_cursor < stack_size) {

                // Pop the top of the stack
                u32 current_node_id = stack(stack_cursor);
                stack(stack_cursor) = _nindex;
                stack_cursor++;

                // check iteraction creteria
                bool cur_id_valid = traverse_condition(current_node_id);

                if (cur_id_valid) { // leaf or cell satisfies the criteria

                    if (is_id_leaf(current_node_id)) { // I found a leaf !!!!!

                        found_leaf = current_node_id;
                        break;

                    } else { // it can interact & not leaf => stack

                        u32 lid = get_left_child(current_node_id);
                        u32 rid = get_right_child(current_node_id);

                        stack(stack_cursor - 1) = rid;
                        stack_cursor--;

                        stack(stack_cursor - 1) = lid;
                        stack_cursor--;
                    }
                } else {
                    // This does not satisfy the criteria => excluded case (gravity for ex.)
                    on_excluded_node(current_node_id);
                }
            }

            if (found_leaf != _nindex) {
                on_found_leaf(found_leaf);
            }
        }
    }

    /// Warp coherent leaf processing variant using a stack supplied by the caller (root = 0)
    template<class StackAccessor, class Functor1, class Functor2, class Functor3>
    inline void stack_based_traversal_leaf_coherent(
        StackAccessor &&stack,
        u32 stack_size,
        Functor1 &&traverse_condition,
        Functor2 &&on_found_leaf,
        Functor3 &&on_excluded_node) const {

        // On a Karras tree, the root is always 0
        u32 root_node = 0;

        stack_based_traversal_leaf_coherent(
            std::forward<StackAccessor>(stack),
            stack_size,
            root_node,
            std::forward<Functor1>(traverse_condition),
            std::forward<Functor2>(on_found_leaf),
            std::forward<Functor3>(on_excluded_node));
    }
};

struct shamtree::KarrasTreeTraverser {

    const sham::DeviceBuffer<u32> &buf_lchild_id;  ///< ref to left child id buffer
    const sham::DeviceBuffer<u32> &buf_rchild_id;  ///< ref to right child id buffer
    const sham::DeviceBuffer<u8> &buf_lchild_flag; ///< ref to left child flag buffer
    const sham::DeviceBuffer<u8> &buf_rchild_flag; ///< ref to right child flag buffer
    u32 offset_leaf; ///< how many internal nodes before the first leaf ?

    /// get read only accessor
    inline KarrasTreeTraverserAccessed get_read_access(sham::EventList &deps) const {
        return KarrasTreeTraverserAccessed{
            .lchild_id   = buf_lchild_id.get_read_access(deps),
            .rchild_id   = buf_rchild_id.get_read_access(deps),
            .lchild_flag = buf_lchild_flag.get_read_access(deps),
            .rchild_flag = buf_rchild_flag.get_read_access(deps),
            .offset_leaf = offset_leaf};
    }

    /// complete the buffer states with the resulting event
    inline void complete_event_state(sycl::event e) const {
        buf_lchild_id.complete_event_state(e);
        buf_rchild_id.complete_event_state(e);
        buf_lchild_flag.complete_event_state(e);
        buf_rchild_flag.complete_event_state(e);
    }

    /// is the root a leaf ?
    inline bool is_root_leaf() const { return offset_leaf == 0; }
};

struct shamtree::KarrasTreeTraverserHost {

    std::vector<u32> buf_lchild_id;  ///< ref to left child id buffer
    std::vector<u32> buf_rchild_id;  ///< ref to right child id buffer
    std::vector<u8> buf_lchild_flag; ///< ref to left child flag buffer
    std::vector<u8> buf_rchild_flag; ///< ref to right child flag buffer
    u32 offset_leaf;                 ///< how many internal nodes before the first leaf ?

    /// get read only accessor
    inline KarrasTreeTraverserAccessed get_read_access() const {
        return KarrasTreeTraverserAccessed{
            .lchild_id   = buf_lchild_id.data(),
            .rchild_id   = buf_rchild_id.data(),
            .lchild_flag = buf_lchild_flag.data(),
            .rchild_flag = buf_rchild_flag.data(),
            .offset_leaf = offset_leaf};
    }
};
