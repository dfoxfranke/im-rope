use proptest::prelude::*;
use proptest::strategy::{NewTree, ValueTree};
use proptest::test_runner::{Reason, TestRunner};
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Copy, Clone)]
enum Instruction {
    Simplify,
    Complicate,
}

#[derive(Clone)]
pub struct Stream<S> {
    strategy: S,
    instructions: Vec<Instruction>,
    runner: TestRunner,
    count: usize,
    seed: Arc<TestRunner>,
    max_count: Arc<AtomicUsize>,
}

impl<S> Stream<S>
where
    S: Strategy,
{
    fn apply_runner(
        strategy: &S,
        runner: &mut TestRunner,
        instructions: &[Instruction],
    ) -> Result<S::Value, Reason> {
        let mut tree = strategy.new_tree(runner)?;
        for i in instructions.iter() {
            match i {
                Instruction::Simplify => tree.simplify(),
                Instruction::Complicate => tree.complicate(),
            };
        }
        Ok(tree.current())
    }

    pub fn try_gen(&mut self) -> Result<S::Value, Reason> {
        self.count += 1;
        self.max_count.fetch_max(self.count, Ordering::Release);
        Self::apply_runner(&self.strategy, &mut self.runner, &self.instructions)
    }

    pub fn gen(&mut self) -> S::Value {
        self.try_gen().unwrap()
    }
}

impl<S> Debug for Stream<S>
where
    S: Strategy,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        struct StreamList<'a, S> {
            strategy: &'a S,
            runner: &'a RefCell<TestRunner>,
            instructions: &'a Vec<Instruction>,
            count: usize,
        }

        impl<'a, S> Debug for StreamList<'a, S>
        where
            S: Strategy,
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                let mut out = f.debug_list();
                for _ in 0..self.count {
                    match Stream::apply_runner(
                        self.strategy,
                        &mut self.runner.borrow_mut(),
                        self.instructions.as_slice(),
                    ) {
                        Ok(entry) => out.entry(&entry),
                        Err(reason) => out.entry(&format!("???{:?}???", &reason)),
                    };
                }
                out.finish()
            }
        }

        let runner = RefCell::new((*self.seed).clone());
        f.debug_struct("Stream")
            .field(
                "past",
                &StreamList {
                    strategy: &self.strategy,
                    runner: &runner,
                    instructions: &self.instructions,
                    count: self.count,
                },
            )
            .field(
                "future",
                &StreamList {
                    strategy: &self.strategy,
                    runner: &runner,
                    instructions: &self.instructions,
                    count: self.max_count.load(Ordering::Acquire) - self.count,
                },
            )
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct StreamTree<S>
where
    S: Strategy,
{
    strategy: S,
    instructions: Vec<Instruction>,
    first: S::Tree,
    seed: Arc<TestRunner>,
    max_count: Arc<AtomicUsize>,
}

impl<S> ValueTree for StreamTree<S>
where
    S: Strategy + Clone,
{
    type Value = Stream<S>;

    fn current(&self) -> Self::Value {
        Stream {
            strategy: self.strategy.clone(),
            instructions: self.instructions.clone(),
            runner: (*self.seed).clone(),
            count: 0,
            seed: self.seed.clone(),
            max_count: self.max_count.clone(),
        }
    }

    fn simplify(&mut self) -> bool {
        let ret = self.first.simplify();
        if ret {
            self.instructions.push(Instruction::Simplify);
            self.max_count = Arc::new(AtomicUsize::new(0));
        }
        ret
    }

    fn complicate(&mut self) -> bool {
        let ret = self.first.complicate();
        if ret {
            self.instructions.push(Instruction::Complicate);
            self.max_count = Arc::new(AtomicUsize::new(0));
        }
        ret
    }
}

#[derive(Debug, Clone)]
pub struct StreamStrategy<S>(pub S);

impl<S> Strategy for StreamStrategy<S>
where
    S: Strategy + Clone,
{
    type Tree = StreamTree<S>;
    type Value = Stream<S>;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let runner = TestRunner::new_with_rng(runner.config().clone(), runner.new_rng());
        let strategy = self.0.clone();
        let first = strategy.new_tree(&mut runner.clone())?;

        Ok(StreamTree {
            strategy,
            instructions: Vec::new(),
            first,
            seed: Arc::new(runner),
            max_count: Arc::new(AtomicUsize::new(0)),
        })
    }
}
