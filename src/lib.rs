use bit_set::BitSet;
use num_traits::{AsPrimitive, Bounded, PrimInt};
use ordered_float::OrderedFloat;
use rand::Rng;
use rand::SeedableRng;
use rustc_hash::FxHashMap;
use std::collections::{BTreeSet, BinaryHeap, HashSet};
use std::f32;
use std::hash::Hash;

use FxHashMap as Dict;

// n.b. this constrains the maximum number index appearances < 256
type Count = u16;
type Score = f32;
type GreedyScore = OrderedFloat<Score>;
type SSAPath = Vec<Vec<u32>>;
// types for optimal optimization
type Subgraph = BitSet;
type BitPath = Vec<(Subgraph, Subgraph)>;
type SubContraction<Ix> = (Vec<(Ix, Count)>, Score, BitPath);

// these allow us write generic implementations for u8, u16, u32
trait IndexType: PrimInt + Bounded + Hash + AsPrimitive<usize> + 'static {}
trait NodeType: PrimInt + Bounded + Hash + AsPrimitive<usize> + AsPrimitive<u32> + 'static {}

impl<T> IndexType for T where T: PrimInt + Bounded + Hash + AsPrimitive<usize> + 'static {}
impl<T> NodeType for T where
    T: PrimInt + Bounded + Hash + AsPrimitive<usize> + AsPrimitive<u32> + 'static
{
}

/// helper struct to build contractions from bottom up
#[derive(Clone)]
struct ContractionProcessor<Ix: IndexType, Node: NodeType> {
    nodes: Dict<Node, Vec<(Ix, Count)>>,
    edges: Dict<Ix, Vec<Node>>,
    appearances: Vec<Count>,
    sizes: Vec<Score>,
    ssa: Node,
    ssa_path: SSAPath,
    track_flops: bool,
    flops: Score,
    flops_limit: Score,
    total_intermediate_size_ln: Score,
    max_intermediate_size_ln: Score,
}

/// given log(x) and log(y) compute log(x + y), without exponentiating both
fn ln_add(lx: Score, ly: Score) -> Score {
    let max_val = lx.max(ly);
    max_val + f32::ln_1p(f32::exp(-f32::abs(lx - ly)))
}

/// given log(x) and log(y) compute log(x - y), without exponentiating both,
/// if (x - y) is negative, return -log(x - y).
fn ln_sub(lx: f32, ly: f32) -> f32 {
    if lx < ly {
        -ly - f32::ln_1p(-f32::exp(lx - ly))
    } else {
        lx + f32::ln_1p(-f32::exp(ly - lx))
    }
}

fn compute_legs<Ix: IndexType>(
    ilegs: &[(Ix, Count)],
    jlegs: &[(Ix, Count)],
    appearances: &Vec<Count>,
) -> Vec<(Ix, Count)> {
    let mut ip = 0;
    let mut jp = 0;
    let ni = ilegs.len();
    let nj = jlegs.len();
    let mut new_legs: Vec<(Ix, Count)> = Vec::with_capacity(ilegs.len() + jlegs.len());

    loop {
        if ip == ni {
            new_legs.extend(jlegs[jp..].iter());
            break;
        }
        if jp == nj {
            new_legs.extend(ilegs[ip..].iter());
            break;
        }

        let (ix, ic) = ilegs[ip];
        let (jx, jc) = jlegs[jp];

        if ix < jx {
            // index only appears in ilegs
            new_legs.push((ix, ic));
            ip += 1;
        } else if ix > jx {
            // index only appears in jlegs
            new_legs.push((jx, jc));
            jp += 1;
        } else {
            // index appears in both
            let new_count = ic + jc;
            if new_count != appearances[ix.as_()] {
                // not last appearance -> kept index contributes to new size
                new_legs.push((ix, new_count));
            }
            ip += 1;
            jp += 1;
        }
    }
    new_legs
}

fn compute_size<Ix: IndexType>(legs: &[(Ix, Count)], sizes: &Vec<Score>) -> Score {
    legs.iter().map(|&(ix, _)| sizes[ix.as_()]).sum()
}

fn compute_flops<Ix: IndexType>(
    ilegs: &[(Ix, Count)],
    jlegs: &[(Ix, Count)],
    sizes: &Vec<Score>,
) -> Score {
    let mut flops: Score = 0.0;
    let mut seen: HashSet<Ix> = HashSet::with_capacity(ilegs.len());
    for &(ix, _) in ilegs {
        seen.insert(ix);
        flops += sizes[ix.as_()];
    }
    for (ix, _) in jlegs {
        if !seen.contains(ix) {
            flops += sizes[ix.as_()];
        }
    }
    flops
}

fn is_simplifiable<Ix: IndexType>(legs: &[(Ix, Count)], appearances: &Vec<Count>) -> bool {
    let mut prev_ix = Ix::max_value();
    for &(ix, ix_count) in legs {
        if (ix == prev_ix) || (ix_count == appearances[ix.as_()]) {
            return true;
        }
        prev_ix = ix;
    }
    false
}

fn compute_simplified<Ix: IndexType>(
    legs: &[(Ix, Count)],
    appearances: &Vec<Count>,
) -> Vec<(Ix, Count)> {
    if legs.len() == 0 {
        return legs.to_vec();
    }
    let mut new_legs: Vec<(Ix, Count)> = Vec::with_capacity(legs.len());

    let (mut cur_ix, mut cur_cnt) = legs[0];
    for &(ix, ix_cnt) in legs.iter().skip(1) {
        if ix == cur_ix {
            // accumulate full stored count (not just +1)
            cur_cnt += ix_cnt;
        } else {
            if cur_cnt != appearances[cur_ix.as_()] {
                new_legs.push((cur_ix, cur_cnt));
            }
            cur_ix = ix;
            cur_cnt = ix_cnt;
        }
    }
    // push final group if not fully consumed
    if cur_cnt != appearances[cur_ix.as_()] {
        new_legs.push((cur_ix, cur_cnt));
    }
    new_legs
}

impl<Ix: IndexType, Node: NodeType> ContractionProcessor<Ix, Node> {
    fn new(
        inputs: Vec<Vec<u32>>,
        output: Vec<u32>,
        size_dict: Dict<u32, f32>,
        track_flops: bool,
    ) -> ContractionProcessor<Ix, Node> {
        if size_dict.len() > Ix::max_value().as_() {
            panic!(
                "cotengrust: too many indices, maximum is {}",
                Ix::max_value().as_()
            );
        }

        let mut nodes: Dict<Node, Vec<(Ix, Count)>> = Dict::default();
    let mut edges: Dict<Ix, Vec<Node>> = Dict::default();
    let mut indmap: Dict<u32, Ix> = Dict::default();
        let mut sizes: Vec<Score> = Vec::with_capacity(size_dict.len());
        let mut appearances: Vec<Count> = Vec::with_capacity(size_dict.len());
        // enumerate index labels as unsigned integers from 0
        let mut c: Ix = Ix::zero();

        for (i, term) in inputs.into_iter().enumerate() {
            let mut legs = Vec::with_capacity(term.len());
            for ind in term {
                match indmap.get(&ind) {
                    None => {
                        // index not parsed yet
                        indmap.insert(ind, c);
                        edges.insert(c, vec![Node::from(i).unwrap()]);
                        appearances.push(1);
                        sizes.push(f32::ln(size_dict[&ind] as f32));
                        legs.push((c, 1));
                        c = c + Ix::one();
                    }
                    Some(&ix) => {
                        // index already present
                        appearances[ix.as_()] += 1;
                        let node = Node::from(i).unwrap();
                        let en = edges.get_mut(&ix).unwrap();
                        if !en.contains(&node) { en.push(node); }
                        legs.push((ix, 1));
                    }
                };
            }
            legs.sort();
            nodes.insert(Node::from(i).unwrap(), legs);
        }
        output.into_iter().for_each(|ind| {
            appearances[indmap[&ind].as_()] += 1;
        });

        let ssa = Node::from(nodes.len()).unwrap();
        let ssa_path: SSAPath = Vec::with_capacity(2 * nodes.len() - 1);
        let flops: Score = 0.0;
        let flops_limit: Score = Score::INFINITY;
    let total_intermediate_size_ln: Score = 0.0;
    let max_intermediate_size_ln: Score = f32::NEG_INFINITY;

        ContractionProcessor {
            nodes,
            edges,
            appearances,
            sizes,
            ssa,
            ssa_path,
            track_flops,
            flops,
            flops_limit,
            total_intermediate_size_ln,
            max_intermediate_size_ln,
        }
    }

    fn neighbors(&self, i: Node) -> Vec<Node> {
        let mut js: Vec<Node> = Vec::new();
        for (ix, _) in self.nodes[&i].iter() {
            if let Some(enodes) = self.edges.get(&ix) {
                for &j in enodes.iter() {
                    if j != i && !js.contains(&j) {
                        js.push(j);
                    }
                }
            }
        }
        js
    }

    /// like neighbors but skip edges with too many neighbors, for greedy
    fn neighbors_limit(&self, i: Node, max_neighbors: usize) -> Vec<Node> {
        let mut js: Vec<Node> = Vec::new();
        for (ix, _) in self.nodes[&i].iter() {
            if let Some(enodes) = self.edges.get(&ix) {
                if max_neighbors != 0 && enodes.len() > max_neighbors {
                    // basically a batch index with too many combinations -> skip
                    continue;
                }
                for &j in enodes.iter() {
                    if j != i && !js.contains(&j) {
                        js.push(j);
                    }
                }
            }
        }
        js
    }

    /// remove an index from the graph, updating all legs
    fn remove_ix(&mut self, ix: Ix) {
        for j in self.edges.remove(&ix).unwrap() {
            self.nodes.get_mut(&j).unwrap().retain(|(k, _)| *k != ix);
        }
    }

    /// remove a node from the graph, update the edgemap, return the legs
    fn pop_node(&mut self, i: Node) -> Vec<(Ix, Count)> {
        let legs = self.nodes.remove(&i).unwrap();
        for (ix, _) in legs.iter() {
            let remove_entire = {
                match self.edges.get_mut(&ix) {
                    Some(enodes) => {
                        if let Some(pos) = enodes.iter().position(|&n| n == i) {
                            enodes.swap_remove(pos);
                        }
                        enodes.is_empty()
                    }
                    None => false,
                }
            };
            if remove_entire {
                self.edges.remove(&ix);
            }
        }
        legs
    }

    /// add a new node to the graph, update the edgemap, return the new id
    fn add_node(&mut self, legs: Vec<(Ix, Count)>) -> Node {
        let i = self.ssa;
        self.ssa = self.ssa + Node::one();
        for (ix, _) in &legs {
            self.edges
                .entry(*ix)
                .and_modify(|nodes| {
                    if !nodes.contains(&i) { nodes.push(i); }
                })
                .or_insert(vec![i]);
        }
        self.nodes.insert(i, legs);
        i
    }

    /// contract two nodes, return the new node id
    fn contract_nodes(&mut self, i: Node, j: Node) -> Node {
        let ilegs = self.pop_node(i);
        let jlegs = self.pop_node(j);
        if self.track_flops {
            let fl = compute_flops(&ilegs, &jlegs, &self.sizes);
            self.flops = ln_add(self.flops, fl);
            // track intermediate sizes (ln)
            let new_legs = compute_legs(&ilegs, &jlegs, &self.appearances);
            let interm_size = compute_size(&new_legs, &self.sizes);
            self.total_intermediate_size_ln = ln_add(self.total_intermediate_size_ln, interm_size);
            if interm_size > self.max_intermediate_size_ln {
                self.max_intermediate_size_ln = interm_size;
            }
        }
        let new_legs = compute_legs(&ilegs, &jlegs, &self.appearances);
        let k = self.add_node(new_legs);
        self.ssa_path.push(vec![i.as_(), j.as_()]);
        k
    }

    /// contract two nodes (which we already know the legs for), return the new node id
    fn contract_nodes_given_legs(&mut self, i: Node, j: Node, new_legs: Vec<(Ix, Count)>) -> Node {
        let ilegs = self.pop_node(i);
        let jlegs = self.pop_node(j);
        if self.track_flops {
            let fl = compute_flops(&ilegs, &jlegs, &self.sizes);
            self.flops = ln_add(self.flops, fl);
            // track intermediate sizes (ln)
            let interm_size = compute_size(&new_legs, &self.sizes);
            self.total_intermediate_size_ln = ln_add(self.total_intermediate_size_ln, interm_size);
            if interm_size > self.max_intermediate_size_ln {
                self.max_intermediate_size_ln = interm_size;
            }
        }
        let k = self.add_node(new_legs);
        self.ssa_path.push(vec![i.as_(), j.as_()]);
        k
    }

    /// find any indices that appear in all terms and just remove/ignore them
    fn simplify_batch(&mut self) {
        let mut ix_to_remove = Vec::new();
        let nterms = self.nodes.len();
        for (ix, ix_nodes) in self.edges.iter() {
            if ix_nodes.len() >= nterms {
                ix_to_remove.push(*ix);
            }
        }
        for ix in ix_to_remove {
            self.remove_ix(ix);
        }
    }

    /// perform any simplifications involving single terms
    fn simplify_single_terms(&mut self) {
        for (i, legs) in self.nodes.clone().into_iter() {
            if is_simplifiable(&legs, &self.appearances) {
                self.pop_node(i);
                let legs_reduced = compute_simplified(&legs, &self.appearances);
                self.add_node(legs_reduced);
                self.ssa_path.push(vec![i.as_()]);
            }
        }
    }

    /// combine and remove all scalars
    fn simplify_scalars(&mut self) {
        let mut scalars = Vec::new();
        let mut j: Option<Node> = None;
        let mut jndim: usize = 0;
        for (i, legs) in self.nodes.iter() {
            let ndim = legs.len();
            if ndim == 0 {
                scalars.push(*i);
            } else {
                // also search for smallest other term to multiply into
                if j.is_none() || ndim < jndim {
                    j = Some(*i);
                    jndim = ndim;
                }
            }
        }
        if !scalars.is_empty() {
            // chain all scalars into a single scalar node
            let mut acc = scalars[0];
            for &s in scalars.iter().skip(1) {
                if self.nodes.contains_key(&acc) && self.nodes.contains_key(&s) {
                    acc = self.contract_nodes(acc, s);
                }
            }
            // multiply resultant scalar into smallest non-scalar (if any) so it doesn't linger
            if let Some(target) = j {
                if self.nodes.contains_key(&acc) && self.nodes.contains_key(&target) {
                    self.contract_nodes(acc, target);
                }
            }
        }
    }

    /// combine all terms that have the same legs
    fn simplify_hadamard(&mut self) {
        // group by full legs (including counts) so only *identical* tensors combine
        let mut groups: Dict<Vec<(Ix, Count)>, Vec<Node>> = Dict::default();
        let mut keys_multi: Vec<Vec<(Ix, Count)>> = Vec::new();
        for (i, legs) in self.nodes.iter() {
            let key = legs.clone(); // legs already sorted
            match groups.get_mut(&key) {
                Some(v) => {
                    if v.len() == 1 { keys_multi.push(key.clone()); }
                    v.push(*i);
                }
                None => {
                    groups.insert(key, vec![*i]);
                }
            }
        }
        for key in keys_multi.into_iter() {
            if let Some(mut group) = groups.remove(&key) {
                while group.len() > 1 {
                    let i = group.pop().unwrap();
                    let j = group.pop().unwrap();
                    let k = self.contract_nodes(i, j);
                    group.push(k);
                }
            }
        }
    }

    /// iteratively perform all simplifications until nothing left to do
    fn simplify(&mut self) {
        self.simplify_batch();
        let mut should_run = true;
        while should_run {
            self.simplify_single_terms();
            self.simplify_scalars();
            let ssa_before = self.ssa;
            self.simplify_hadamard();
            should_run = ssa_before != self.ssa;
        }
    }

    /// find disconnected subgraphs
    fn subgraphs(&self) -> SSAPath {
        let mut remaining: BTreeSet<Node> = BTreeSet::default();
        self.nodes.keys().for_each(|i| {
            remaining.insert(*i);
        });
        let mut groups: SSAPath = Vec::new();
        while remaining.len() > 0 {
            let i = remaining.pop_first().unwrap();
            let mut queue: Vec<Node> = vec![i];
            let mut group: BTreeSet<Node> = vec![i].into_iter().collect();
            while queue.len() > 0 {
                let i = queue.pop().unwrap();
                for j in self.neighbors(i) {
                    if !group.contains(&j) {
                        group.insert(j);
                        queue.push(j);
                    }
                }
            }
            group.iter().for_each(|i| {
                remaining.remove(i);
            });
            groups.push(group.into_iter().map(|n| n.as_()).collect());
        }
        groups
    }

    /// greedily optimize the contraction order of all terms
    fn optimize_greedy(
        &mut self,
        costmod: Option<f32>,
        temperature: Option<f32>,
        max_neighbors: Option<usize>,
        seed: Option<u64>,
    ) -> bool {
        let coeff_t = temperature.unwrap_or(0.0);
        let log_coeff_a = f32::ln(costmod.unwrap_or(1.0));
        let max_neighbors = max_neighbors.unwrap_or(16);

        let mut rng = if coeff_t != 0.0 {
            Some(match seed {
                Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
                None => rand::rngs::StdRng::from_os_rng(),
            })
        } else {
            // zero temp - no need for rng
            None
        };

        let mut local_score = |sa: Score, sb: Score, sab: Score| -> Score {
            let gumbel = if let Some(rng) = &mut rng {
                coeff_t * -f32::ln(-f32::ln(rng.random()))
            } else {
                0.0 as f32
            };
            ln_sub(sab - log_coeff_a, ln_add(sa, sb) + log_coeff_a) - gumbel
        };

        // cache all current nodes sizes as we go
        let mut node_sizes: Dict<Node, Score> = Dict::default();
        self.nodes.iter().for_each(|(&i, legs)| {
            node_sizes.insert(i, compute_size(&legs, &self.sizes));
        });

        // we will *deincrement* c, since its a max-heap
        let mut c: i32 = 0;
        let mut queue: BinaryHeap<(GreedyScore, i32)> =
            BinaryHeap::with_capacity(self.edges.len() * 2);

        // the heap keeps a reference to actual contraction info in this
        let mut contractions: Dict<i32, (Node, Node, Score, Vec<(Ix, Count)>)> = Dict::default();

        // get the initial candidate contractions
        for ix_nodes in self.edges.values() {
            if max_neighbors != 0 && ix_nodes.len() > max_neighbors {
                // basically a batch index with too many combinations -> skip
                continue;
            }

            // convert to vector for combinational indexing
            let ix_nodes: Vec<Node> = ix_nodes.iter().cloned().collect();
            // for all combinations of nodes with a connected edge
            for ip in 0..ix_nodes.len() {
                let i = ix_nodes[ip];
                let isize = node_sizes[&i];
                for jp in (ip + 1)..ix_nodes.len() {
                    let j = ix_nodes[jp];
                    let jsize = node_sizes[&j];
                    let klegs = compute_legs(&self.nodes[&i], &self.nodes[&j], &self.appearances);
                    let ksize = compute_size(&klegs, &self.sizes);
                    let score = local_score(isize, jsize, ksize);
                    queue.push((OrderedFloat(-score), c));
                    contractions.insert(c, (i, j, ksize, klegs));
                    c -= 1;
                }
            }
        }

        // greedily contract remaining
        let mut steps: usize = 0;
        while let Some((_, c0)) = queue.pop() {
            let (i, j, ksize, klegs) = contractions.remove(&c0).unwrap();
            if !self.nodes.contains_key(&i) || !self.nodes.contains_key(&j) {
                // one of the nodes has been removed -> skip
                continue;
            }

            // perform contraction:
            let k = self.contract_nodes_given_legs(i, j, klegs.clone());

            if self.track_flops && self.flops >= self.flops_limit {
                // stop if we have reached the flops limit
                return false;
            }

            node_sizes.insert(k, ksize);

            for l in self.neighbors_limit(k, max_neighbors) {
                // assess all neighboring contractions of new node
                let llegs = &self.nodes[&l];
                let lsize = node_sizes[&l];
                // get candidate legs and size
                let mlegs = compute_legs(&klegs, llegs, &self.appearances);
                let msize = compute_size(&mlegs, &self.sizes);
                let score = local_score(ksize, lsize, msize);
                queue.push((OrderedFloat(-score), c));
                contractions.insert(c, (k, l, msize, mlegs));
                c -= 1;
            }

            // periodically prune queue and contractions to remove stale entries
            steps += 1;
            if steps % 4096 == 0 {
                let mut valid = Vec::new();
                for (score, cid) in queue.drain() {
                    if let Some((i, j, _, _)) = contractions.get(&cid) {
                        if self.nodes.contains_key(&i) && self.nodes.contains_key(&j) {
                            valid.push((score, cid));
                        } else {
                            contractions.remove(&cid);
                        }
                    }
                }
                queue = BinaryHeap::from(valid);
            }

            // // potential queue pruning?
            // if queue.len() > 100_000 {
            //     let mut valid_contractions = Vec::new();
            //     for (score, cid) in queue.drain() {
            //         if let Some((i, j, _, _)) = contractions.get(&cid) {
            //             if self.nodes.contains_key(&i) && self.nodes.contains_key(&j) {
            //                 valid_contractions.push((score, cid));
            //             } else {
            //                 // Remove stale contraction from map
            //                 contractions.remove(&cid);
            //             }
            //         }
            //     }
            //     queue = BinaryHeap::from(valid_contractions);
            // }
        }
        // success
        return true;
    }

    /// Optimize the contraction order of all terms using a greedy algorithm
    /// that contracts the smallest two terms. Typically only called once
    /// only disconnected subgraph terms (outer products) remain.
    fn optimize_remaining_by_size(&mut self) {
        if self.nodes.len() == 1 {
            // nothing to do
            return;
        };

        let mut nodes_sizes: BinaryHeap<(GreedyScore, Node)> = BinaryHeap::default();
        self.nodes.iter().for_each(|(node, legs)| {
            nodes_sizes.push((OrderedFloat(-compute_size(&legs, &self.sizes)), *node));
        });

        let (_, mut i) = nodes_sizes.pop().unwrap();
        let (_, mut j) = nodes_sizes.pop().unwrap();
        let mut k = self.contract_nodes(i, j);

        while self.nodes.len() > 1 {
            // contract the smallest two nodes until only one remains
            let ksize = compute_size(&self.nodes[&k], &self.sizes);
            nodes_sizes.push((OrderedFloat(-ksize), k));
            (_, i) = nodes_sizes.pop().unwrap();
            (_, j) = nodes_sizes.pop().unwrap();
            k = self.contract_nodes(i, j);
        }
    }
}

fn single_el_bitset(x: usize, n: usize) -> BitSet {
    let mut a = BitSet::with_capacity(n);
    a.insert(x);
    a
}

fn compute_con_cost_flops<Ix: IndexType>(
    temp_legs: Vec<(Ix, Count)>,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    _factor: Score,
) -> (Vec<(Ix, Count)>, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Vec<(Ix, Count)> = Vec::with_capacity(temp_legs.len());
    let mut cost: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        // all involved indices contribute to the cost
        let d = sizes[ix.as_()];
        cost += d;
        if ix_count != appearances[ix.as_()] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
        }
    }
    let new_score = ln_add(ln_add(iscore, jscore), cost);
    (new_legs, new_score)
}

fn compute_con_cost_max<Ix: IndexType>(
    temp_legs: Vec<(Ix, Count)>,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    _factor: Score,
) -> (Vec<(Ix, Count)>, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Vec<(Ix, Count)> = Vec::with_capacity(temp_legs.len());
    let mut cost: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        // all involved indices contribute to the cost
        let d = sizes[ix.as_()];
        cost += d;
        if ix_count != appearances[ix.as_()] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
        }
    }
    let new_score = iscore.max(jscore).max(cost);
    (new_legs, new_score)
}

fn compute_con_cost_size<Ix: IndexType>(
    temp_legs: Vec<(Ix, Count)>,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    _factor: Score,
) -> (Vec<(Ix, Count)>, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Vec<(Ix, Count)> = Vec::with_capacity(temp_legs.len());
    let mut size: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        if ix_count != appearances[ix.as_()] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
            size += sizes[ix.as_()];
        }
    }
    let new_score = iscore.max(jscore).max(size);
    (new_legs, new_score)
}

fn compute_con_cost_write<Ix: IndexType>(
    temp_legs: Vec<(Ix, Count)>,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    _factor: Score,
) -> (Vec<(Ix, Count)>, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Vec<(Ix, Count)> = Vec::with_capacity(temp_legs.len());
    let mut size: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        if ix_count != appearances[ix.as_()] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
            size += sizes[ix.as_()];
        }
    }
    let new_score = ln_add(ln_add(iscore, jscore), size);
    (new_legs, new_score)
}

fn compute_con_cost_combo<Ix: IndexType>(
    temp_legs: Vec<(Ix, Count)>,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    factor: Score,
) -> (Vec<(Ix, Count)>, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Vec<(Ix, Count)> = Vec::with_capacity(temp_legs.len());
    let mut size: Score = 0.0;
    let mut cost: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        // all involved indices contribute to the cost
        let d = sizes[ix.as_()];
        cost += d;
        if ix_count != appearances[ix.as_()] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
            size += d;
        }
    }
    // the score just for this contraction
    let new_local_score = ln_add(cost, factor + size);

    // the total score including history
    let new_score = ln_add(ln_add(iscore, jscore), new_local_score);

    (new_legs, new_score)
}

fn compute_con_cost_limit<Ix: IndexType>(
    temp_legs: Vec<(Ix, Count)>,
    appearances: &Vec<Count>,
    sizes: &Vec<Score>,
    iscore: Score,
    jscore: Score,
    factor: Score,
) -> (Vec<(Ix, Count)>, Score) {
    // remove indices that have reached final appearance
    // and compute cost and size of local contraction
    let mut new_legs: Vec<(Ix, Count)> = Vec::with_capacity(temp_legs.len());
    let mut size: Score = 0.0;
    let mut cost: Score = 0.0;
    for (ix, ix_count) in temp_legs.into_iter() {
        // all involved indices contribute to the cost
        let d = sizes[ix.as_()];
        cost += d;
        if ix_count != appearances[ix.as_()] {
            // not last appearance -> kept index contributes to new size
            new_legs.push((ix, ix_count));
            size += d;
        }
    }
    // whichever is more expensive, the cost or the scaled write
    let new_local_score = cost.max(factor + size);

    // the total score including history
    let new_score = ln_add(ln_add(iscore, jscore), new_local_score);

    (new_legs, new_score)
}

impl<Ix: IndexType, Node: NodeType> ContractionProcessor<Ix, Node> {
    fn optimize_optimal_connected(
        &mut self,
        subgraph: Vec<u32>,
        minimize: Option<String>,
        cost_cap: Option<Score>,
        search_outer: Option<bool>,
    ) {
        // parse the minimize argument
        let minimize = minimize.unwrap_or("flops".to_string());
        let mut minimize_split = minimize.split('-');
        let minimize_type = minimize_split.next().unwrap();
        let factor = minimize_split
            .next()
            .map_or(64.0, |s| s.parse::<f32>().unwrap())
            .ln();
        if minimize_split.next().is_some() {
            // multiple hyphens -> raise error
            panic!("invalid minimize: {:?}", minimize);
        }
        let compute_cost = match minimize_type {
            "flops" => compute_con_cost_flops::<Ix>,
            "max" => compute_con_cost_max::<Ix>,
            "size" => compute_con_cost_size::<Ix>,
            "write" => compute_con_cost_write::<Ix>,
            "combo" => compute_con_cost_combo::<Ix>,
            "limit" => compute_con_cost_limit::<Ix>,
            _ => panic!(
                "minimize must be one of 'flops', 'max', 'size', 'write', 'combo', or 'limit', got {}",
                minimize
            ),
        };
        let search_outer = search_outer.unwrap_or(false);

        // storage for each possible contraction to reach subgraph of size m
        let mut contractions: Vec<Dict<Subgraph, SubContraction<Ix>>> =
            vec![Dict::default(); subgraph.len() + 1];
        // intermediate storage for the entries we are expanding
        let mut contractions_m_temp: Vec<(Subgraph, SubContraction<Ix>)> = Vec::new();
        // need to keep these separately
        let mut best_scores: Dict<Subgraph, Score> = Dict::default();

        // we use linear index within terms given during optimization, this maps
        // back to the original node index
        let nterms = subgraph.len();
        let mut termmap: Dict<Subgraph, Node> = Dict::default();

        for (i, node_u32) in subgraph.into_iter().enumerate() {
            let node = Node::from(node_u32 as usize).unwrap();
            let isubgraph = single_el_bitset(i, nterms);
            termmap.insert(isubgraph.clone(), node);
            let ilegs = self.nodes[&node].clone();
            let iscore: Score = 0.0;
            let ipath: BitPath = Vec::new();
            contractions[1].insert(isubgraph, (ilegs, iscore, ipath));
        }

        let mut ip: usize;
        let mut jp: usize;
        let mut skip_because_outer: bool;

        let cost_cap_incr = f32::ln(2.0);
        let mut cost_cap = cost_cap.unwrap_or(cost_cap_incr);
        while contractions[nterms].len() == 0 {
            // try building subgraphs of size m
            for m in 2..=nterms {
                // out of bipartitions of size (k, m - k)
                for k in 1..=m / 2 {
                    for (isubgraph, (ilegs, iscore, ipath)) in contractions[k].iter() {
                        for (jsubgraph, (jlegs, jscore, jpath)) in contractions[m - k].iter() {
                            // filter invalid combinations first
                            if !isubgraph.is_disjoint(&jsubgraph) || {
                                (k == m - k) && isubgraph.gt(&jsubgraph)
                            } {
                                // subgraphs overlap -> not valid, or
                                // equal subgraph size -> only process sorted pairs
                                continue;
                            }

                            let mut temp_legs: Vec<(Ix, Count)> =
                                Vec::with_capacity(ilegs.len() + jlegs.len());
                            ip = 0;
                            jp = 0;
                            // if search_outer -> we will never skip
                            skip_because_outer = !search_outer;
                            while ip < ilegs.len() && jp < jlegs.len() {
                                if ilegs[ip].0 < jlegs[jp].0 {
                                    // index only appears in ilegs
                                    temp_legs.push(ilegs[ip]);
                                    ip += 1;
                                } else if ilegs[ip].0 > jlegs[jp].0 {
                                    // index only appears in jlegs
                                    temp_legs.push(jlegs[jp]);
                                    jp += 1;
                                } else {
                                    // index appears in both
                                    temp_legs.push((ilegs[ip].0, ilegs[ip].1 + jlegs[jp].1));
                                    ip += 1;
                                    jp += 1;
                                    skip_because_outer = false;
                                }
                            }
                            if skip_because_outer {
                                // no shared indices -> outer product
                                continue;
                            }
                            // add any remaining indices
                            temp_legs.extend(ilegs[ip..].iter().chain(jlegs[jp..].iter()));

                            // compute candidate contraction result and score
                            let (new_legs, new_score) = compute_cost(
                                temp_legs,
                                &self.appearances,
                                &self.sizes,
                                *iscore,
                                *jscore,
                                factor,
                            );

                            if new_score > cost_cap {
                                // contraction not allowed yet due to 'sieve'
                                continue;
                            }

                            // check candidate against current best subgraph path
                            let new_subgraph: Subgraph = isubgraph.union(&jsubgraph).collect();

                            // because we have to do a delayed update of
                            // contractions[m] for borrowing reasons, we check
                            // against a non-delayed score lookup so we don't
                            // overwrite best scores within the same iteration
                            let found_new_best = match best_scores.get(&new_subgraph) {
                                Some(current_score) => new_score < *current_score,
                                None => true,
                            };
                            if found_new_best {
                                best_scores.insert(new_subgraph.clone(), new_score);
                                // only need the path if updating
                                let mut new_path: BitPath =
                                    Vec::with_capacity(ipath.len() + jpath.len() + 1);
                                new_path.extend_from_slice(&ipath);
                                new_path.extend_from_slice(&jpath);
                                new_path.push((isubgraph.clone(), jsubgraph.clone()));
                                contractions_m_temp
                                    .push((new_subgraph, (new_legs, new_score, new_path)));
                            }
                        }
                    }
                    // move new contractions from temp into the main storage,
                    // there might be contractions for the same subgraph in
                    // this, but because we check eagerly best_scores above,
                    // later entries are guaranteed to be better
                    contractions_m_temp.drain(..).for_each(|(k, v)| {
                        contractions[m].insert(k, v);
                    });
                }
            }
            cost_cap += cost_cap_incr;
        }
        // can only ever be a single entry in contractions[nterms] -> the best
        let (_, _, best_path) = contractions[nterms].values().next().unwrap();

        // convert from the bitpath to the actual (subgraph) node ids
        for (isubgraph, jsubgraph) in best_path.into_iter() {
            let i = termmap[&isubgraph];
            let j = termmap[&jsubgraph];
            let k = self.contract_nodes(i, j);
            let ksubgraph: Subgraph = isubgraph.union(&jsubgraph).collect();
            termmap.insert(ksubgraph, k);
        }
    }

    fn optimize_optimal(
        &mut self,
        minimize: Option<String>,
        cost_cap: Option<Score>,
        search_outer: Option<bool>,
    ) {
        for subgraph in self.subgraphs() {
            self.optimize_optimal_connected(subgraph, minimize.clone(), cost_cap, search_outer);
        }
    }
}

// ----------------------- dispatch-able functions ------------------------- //

fn run_find_subgraphs<Ix: IndexType, Node: NodeType>(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Dict<u32, f32>,
) -> SSAPath {
    let cp: ContractionProcessor<Ix, Node> =
        ContractionProcessor::new(inputs, output, size_dict, false);
    cp.subgraphs()
}

fn run_simplify<Ix: IndexType, Node: NodeType>(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Dict<u32, f32>,
) -> SSAPath {
    let mut cp: ContractionProcessor<Ix, Node> =
        ContractionProcessor::new(inputs, output, size_dict, false);
    cp.simplify();
    cp.ssa_path
}

fn run_greedy<Ix: IndexType, Node: NodeType>(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Dict<u32, f32>,
    costmod: Option<f32>,
    temperature: Option<f32>,
    max_neighbors: Option<usize>,
    seed: Option<u64>,
    simplify: bool,
) -> SSAPath {
    let mut cp: ContractionProcessor<Ix, Node> =
        ContractionProcessor::new(inputs, output, size_dict, false);
    if simplify {
        cp.simplify();
    }
    cp.optimize_greedy(costmod, temperature, max_neighbors, seed);
    cp.optimize_remaining_by_size();
    cp.ssa_path
}

fn run_optimal<Ix: IndexType, Node: NodeType>(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Dict<u32, f32>,
    minimize: Option<String>,
    cost_cap: Option<Score>,
    search_outer: Option<bool>,
    simplify: bool,
) -> SSAPath {
    let mut cp: ContractionProcessor<Ix, Node> =
        ContractionProcessor::new(inputs, output, size_dict, false);
    if simplify {
        cp.simplify();
    }
    cp.optimize_optimal(minimize, cost_cap, search_outer);
    cp.optimize_remaining_by_size();
    cp.ssa_path
}

/// helper function for random greedy optimization with flop tracking
#[allow(dead_code)]
fn run_random_greedy_optimization<Ix: IndexType, Node: NodeType>(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Dict<u32, f32>,
    simplify: bool,
    seeds: &[u64],
    costmod_min: f32,
    costmod_diff: f32,
    is_const_costmod: bool,
    temp_min: f32,
    log_temp_min: f32,
    log_temp_diff: f32,
    is_const_temp: bool,
    max_neighbors: Option<usize>,
    rng: &mut rand::rngs::StdRng,
) -> (SSAPath, Score) {
    let mut cp0: ContractionProcessor<Ix, Node> =
        ContractionProcessor::new(inputs, output, size_dict, true);
    if simplify {
        cp0.simplify();
    }

    let mut best_path: Option<SSAPath> = None;
    let mut best_flops = f32::INFINITY;

    for seed in seeds.iter() {
        let mut cp = cp0.clone();

        let costmod = if is_const_costmod {
            costmod_min
        } else {
            costmod_min + rng.random::<f32>() * costmod_diff
        };

        let temperature = if is_const_temp {
            temp_min
        } else {
            f32::exp(log_temp_min + rng.random::<f32>() * log_temp_diff)
        };

        let success =
            cp.optimize_greedy(Some(costmod), Some(temperature), max_neighbors, Some(*seed));

        if !success {
            continue;
        }

        cp.optimize_remaining_by_size();

        if cp.flops < best_flops {
            best_path = Some(cp.ssa_path);
            best_flops = cp.flops;
            cp0.flops_limit = cp.flops;
        }
    }

    best_flops *= f32::consts::LOG10_E;
    (best_path.unwrap(), best_flops)
}

// --------------------------- WOLFRAM FUNCTIONS ---------------------------- //

use wolfram_library_link as wll;


wll::generate_loader!(load_wstp_functions);


fn ssa_to_linear(ssa_path: SSAPath, n: Option<usize>) -> SSAPath {
    let n = match n {
        Some(n) => n,
        None => ssa_path.iter().map(|v| v.len()).sum::<usize>() + ssa_path.len() + 1,
    };
    let mut ids: Vec<u32> = (0..n).map(|i| i as u32).collect();
    let mut path: SSAPath = Vec::with_capacity(2 * n - 1);
    let mut ssa = n as u32;
    for scon in ssa_path {
        // find the locations of the ssa ids in the list of ids
        let mut con: Vec<u32> = scon
            .iter()
            .map(|s| ids.binary_search(s).unwrap() as u32)
            .collect();
        // remove the ssa ids from the list
        con.sort();
        for j in con.iter().rev() {
            ids.remove(*j as usize);
        }
        path.push(con);
        ids.push(ssa);
        ssa += 1;
    }
    path
}

#[cfg(test)]
mod perf_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bench_user_dataset_costs() {
        // Small example dataset using u32 labels directly
        let inputs: Vec<Vec<u32>> = vec![
            vec![3], vec![1], vec![4], vec![2], vec![5], vec![7,1,2],
            vec![9,6,8,3,4,5], vec![10,11,6,7,8],
            vec![12,13,14,16,15,9,10,11], vec![17,18,12,13,14,15],
            vec![24,20,16], vec![19,17], vec![21,22,23,25,18,19,20],
            vec![31,26,29,35,30,21,22,23,24,25], vec![27,26],
            vec![28,27], vec![33,36,28,29,30], vec![32,34,31],
            vec![37,38,39,32,33,34,35,36], vec![40,41,42,37],
        ];

        let output: Vec<u32> = vec![40,41,42,38,39];

        let mut sizes: Dict<u32, f32> = Dict::default();
        let sizes_list: &[(u32, i32)] = &[
            (3,4),(1,4),(4,2),(2,4),(5,3),(7,4),(9,4),(6,2),(8,3),(10,4),(11,3),(12,4),(13,4),(14,2),(16,4),(15,3),(17,4),(18,4),(24,4),(20,3),(19,2),(21,4),(22,4),(23,2),(25,3),(31,4),(26,4),(29,2),(35,4),(30,3),(27,4),(28,4),(33,4),(36,3),(32,4),(34,2),(37,2),(38,4),(39,3),(40,4),(41,4),(42,2)
        ];
        for (k, v) in sizes_list.iter() { sizes.insert(*k, *v as f32); }

        // We'll collect results and print an aligned ASCII table
        struct Row {
            method: &'static str,
            minimize: String,
            time_ms: f64,
            flops: f32,
            total_inter: f32,
            max_inter: f32,
            path: String,
        }

        let mut rows: Vec<Row> = Vec::new();

    // run greedy once and collect (use u32 index/node types)
    let mut cp_g: ContractionProcessor<u32,u32> = ContractionProcessor::new(inputs.clone(), output.clone(), sizes.clone(), true);
        cp_g.simplify();
        let t0 = Instant::now();
        let _ = cp_g.optimize_greedy(None, None, None, None);
        cp_g.optimize_remaining_by_size();
        let dt = t0.elapsed();
    let greedy_flops = f32::exp(cp_g.flops);
    let total_inter = f32::exp(cp_g.total_intermediate_size_ln);
    let max_inter = if cp_g.max_intermediate_size_ln.is_finite() { f32::exp(cp_g.max_intermediate_size_ln) } else { 0.0 };
        // obtain linear SSA path string for greedy
        let greedy_path_ssa = ssa_to_linear(cp_g.ssa_path.clone(), Some(inputs.len()));
        let greedy_path_str = format!("{:?}", greedy_path_ssa);
        rows.push(Row {
            method: "greedy",
            minimize: "-".to_string(),
            time_ms: dt.as_secs_f64() * 1000.0,
            flops: greedy_flops,
            total_inter: total_inter,
            max_inter: max_inter,
            path: greedy_path_str,
        });

        // try a few minimize options for optimal and collect
        let minimizes: Vec<Option<String>> = vec![None, Some("flops".to_string()), Some("max".to_string()), Some("size".to_string()), Some("write".to_string()), Some("combo".to_string()), Some("limit-64".to_string())];
        for m in minimizes.into_iter() {
            let mut cp_o: ContractionProcessor<u32,u32> = ContractionProcessor::new(inputs.clone(), output.clone(), sizes.clone(), true);
            cp_o.simplify();
            let t1 = Instant::now();
            cp_o.optimize_optimal(m.clone(), None, None);
            cp_o.optimize_remaining_by_size();
            let dt_o = t1.elapsed();
            let opt_flops = f32::exp(cp_o.flops);
            let total_inter_ln_o = cp_o.total_intermediate_size_ln;
            let total_inter_o = f32::exp(total_inter_ln_o);
            let max_inter_o = if cp_o.max_intermediate_size_ln.is_finite() { f32::exp(cp_o.max_intermediate_size_ln) } else { 0.0 };
            let opt_path_ssa = ssa_to_linear(cp_o.ssa_path.clone(), Some(inputs.len()));
            let opt_path_str = format!("{:?}", opt_path_ssa);
            rows.push(Row {
                method: "optimal",
                minimize: match m.clone() { Some(s) => s, None => "None".to_string() },
                time_ms: dt_o.as_secs_f64() * 1000.0,
                flops: opt_flops,
                total_inter: total_inter_o,
                max_inter: max_inter_o,
                path: opt_path_str,
            });
        }
        // pretty-print table header and rows with fixed column widths
        println!("{:<8} | {:<10} | {:>10} | {:>15} | {:>15} | {:>15} | {}",
            "Method", "Minimize", "time_ms", "flops", "total_inter", "max_inter", "path");
        // compute separator length from header string for robustness
        let header = format!("{:<8} | {:<10} | {:>10} | {:>15} | {:>15} | {:>15} | {}",
            "Method", "Minimize", "time_ms", "flops", "total_inter", "max_inter", "path");
        println!("{}", "-".repeat(header.len()));
        for r in rows.iter() {
            println!("{:<8} | {:<10} | {:10.3} | {:15.0} | {:15.0} | {:15.0} | {}",
                r.method, r.minimize, r.time_ms, r.flops, r.total_inter, r.max_inter, r.path);
        }
    }
}

#[wll::export]
fn find_subgraphs(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Dict<u32, f32>,
) -> SSAPath {
    let num_indices = size_dict.len();
    let max_nodes = 2 * inputs.len();

    let subgraphs = match (num_indices, max_nodes) {
        (idx, nodes) if idx <= u8::MAX as usize && nodes <= u8::MAX as usize => {
            run_find_subgraphs::<u8, u8>(inputs, output, size_dict)
        }
        (idx, nodes) if idx <= u16::MAX as usize && nodes <= u16::MAX as usize => {
            run_find_subgraphs::<u16, u16>(inputs, output, size_dict)
        }
        _ => run_find_subgraphs::<u32, u32>(inputs, output, size_dict),
    };

    subgraphs
}


#[wll::export]
fn optimize_simplify(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Dict<u32, f32>,
    use_ssa: Option<bool>,
) -> SSAPath {
    let n = inputs.len();
    let num_indices = size_dict.len();
    let max_nodes = 2 * n;

    let ssa_path = match (num_indices, max_nodes) {
        (idx, nodes) if idx <= u8::MAX as usize && nodes <= u8::MAX as usize => {
            run_simplify::<u8, u8>(inputs, output, size_dict)
        }
        (idx, nodes) if idx <= u16::MAX as usize && nodes <= u16::MAX as usize => {
            run_simplify::<u16, u16>(inputs, output, size_dict)
        }
        _ => run_simplify::<u32, u32>(inputs, output, size_dict),
    };

    if use_ssa.unwrap_or(false) {
        ssa_path
    } else {
        ssa_to_linear(ssa_path, Some(n))
    }
}

#[wll::export]
fn optimize_greedy(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Dict<u32, f32>,
    costmod: Option<f32>,
    temperature: Option<f32>,
    max_neighbors: Option<usize>,
    seed: Option<u64>,
    simplify: Option<bool>,
    use_ssa: Option<bool>,
) -> SSAPath {
    let n = inputs.len();
    let num_indices = size_dict.len();
    let max_nodes = 2 * n;
    let simplify = simplify.unwrap_or(true);

    let ssa_path = match (num_indices, max_nodes) {
        (idx, nodes) if idx <= u8::MAX as usize && nodes <= u8::MAX as usize => {
            run_greedy::<u8, u8>(
                inputs,
                output,
                size_dict,
                costmod,
                temperature,
                max_neighbors,
                seed,
                simplify,
            )
        }
        (idx, nodes) if idx <= u16::MAX as usize && nodes <= u16::MAX as usize => {
            run_greedy::<u16, u16>(
                inputs,
                output,
                size_dict,
                costmod,
                temperature,
                max_neighbors,
                seed,
                simplify,
            )
        }
        _ => run_greedy::<u32, u32>(
            inputs,
            output,
            size_dict,
            costmod,
            temperature,
            max_neighbors,
            seed,
            simplify,
        ),
    };

    if use_ssa.unwrap_or(false) {
        ssa_path
    } else {
        ssa_to_linear(ssa_path, Some(n))
    }
}


#[wll::export]
fn optimize_optimal(
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    size_dict: Dict<u32, f32>,
    minimize: Option<String>,
    cost_cap: Option<Score>,
    search_outer: Option<bool>,
    simplify: Option<bool>,
    use_ssa: Option<bool>,
) -> SSAPath {
    let n = inputs.len();
    let num_indices = size_dict.len();
    let max_nodes = 2 * n;
    let simplify = simplify.unwrap_or(true);

    let ssa_path = match (num_indices, max_nodes) {
        (idx, nodes) if idx <= u8::MAX as usize && nodes <= u8::MAX as usize => {
            run_optimal::<u8, u8>(
                inputs,
                output,
                size_dict,
                minimize,
                cost_cap,
                search_outer,
                simplify,
            )
        }
        (idx, nodes) if idx <= u16::MAX as usize && nodes <= u16::MAX as usize => {
            run_optimal::<u16, u16>(
                inputs,
                output,
                size_dict,
                minimize,
                cost_cap,
                search_outer,
                simplify,
            )
        }
        _ => run_optimal::<u32, u32>(
            inputs,
            output,
            size_dict,
            minimize,
            cost_cap,
            search_outer,
            simplify,
        ),
    };

    if use_ssa.unwrap_or(false) {
        ssa_path
    } else {
        ssa_to_linear(ssa_path, Some(n))
    }
}

