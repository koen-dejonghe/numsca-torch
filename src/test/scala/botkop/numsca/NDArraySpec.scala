package botkop.numsca

import org.scalatest.{FlatSpec, Matchers}

import scala.concurrent.{Await, Future}
import scala.language.postfixOps

class NDArraySpec extends FlatSpec with Matchers {

  "An ND array" should "create with provided data" in {

    val data = (1 to 6).toArray.map(_.toFloat)
    val shape = List(2, 3)
    val a = NDArray.create(data, shape)

    a.dim shouldBe 2
    a.shape shouldBe List(2, 3)
    a.size shouldBe 6
    a.data shouldBe Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
  }

  it should "make a zero array" in {
    val shape = List(2, 3)
    val z = NDArray.zeros(shape)
    z.shape shouldBe List(2, 3)
    z.data shouldBe Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  }

  it should "free when no longer used" in {
    val t3 = NDArray.ones(List(100, 100)) // this one will only get garbage collected at the end of the program
    for (_ <- 1 to 100) {
      NDArray.zeros(List(3000, 3000)) // these will get GC'ed as soon as as System.gc() is called
      Thread.sleep(1)
    }
    t3.desc shouldBe "torch.xTensor of size 100x100"
    t3.data.sum shouldBe 100 * 100
  }

  it should "free in parallel when no longer used" in {
    import scala.concurrent.ExecutionContext.Implicits.global
    import scala.concurrent.duration._

    val t3 = NDArray.ones(100, 100) // this one will only get garbage collected at the end of the program

    val futures = Future.sequence {
      (1 to 100).map { _ =>
        Future {
          NDArray.zeros(3000, 3000) // these will get GC'ed as soon as as System.gc() is called
          Thread.sleep(10)
        }
      }
    }

    Await.result(futures, 10 seconds)
    t3.desc shouldBe "torch.xTensor of size 100x100"
    t3.data.sum shouldBe 100 * 100
  }

  it should "arange" in {
    val t = NDArray.arange(max = 10.0)
    t.shape shouldBe List(10)
    t.data shouldBe Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    val u = NDArray.arange(max = 10.0, step = 2.5)
    u.shape shouldBe List(4)
    u.data shouldBe Array(0.0, 2.5, 5.0, 7.5)
  }

  it should "seed" in {
    NDArray.setSeed(213L)
    val t = NDArray.randn(2, 3)
    println(t.data.toList)
    // hard to test
  }

  it should "randn" in {
    NDArray.setSeed(213L)
    val t = NDArray.randn(1000)
    t.shape shouldBe List(1000)
    val data = t.data
    val mean = data.sum / data.length
    println(mean)
    mean should be(0.0f +- 1e-1f)
    val std =
      Math.sqrt(data.map(d => Math.pow(d - mean, 2.0)).sum / (data.length - 1))
    println(std)
    std should be(1.0 +- 1e-1)
  }

  it should "randint" in {
    NDArray.setSeed(213L)
    val t = NDArray.randint(high = 10.0, shape = List(10))
    val data = t.data
    println(data.toList)
  }

  it should "linspace" in {
    val steps = 5
    val t = NDArray.linspace(0, 1, steps)
    t.shape shouldBe List(steps)
    val data = t.data
    data shouldBe Array(0.0, 0.25, 0.5, 0.75, 1.0)
  }

  //--------------------
  it should "cmul" in {
    val t1 = NDArray.arange(1, 10)
    val t2 = NDArray.arange(2, 11)
    val t3 = NDArray.cmul(t2, t1)
    println(t3.shape)
    println(t3.data.toList)
  }

  it should "reshape" in {
    val a = NDArray.arange(max = 9)
    println(a.shape)
    val b = a.reshape(List(3, 3))
    println(b.shape)

  }

  it should "select" in {
    val a = NDArray.arange(max = 9).reshape(3, 3)
    val b = NDArray.select(a, dimension = 0, sliceIndex = 1)
    println(b.shape)
    println(b.data.toList)

  }

  it should "select 2" in {
    val a = NDArray.arange(max = 8).reshape(2, 2, 2)
    // val r = NDArray.select(a, List(0, 1, 0))
    val r = a(0, 1, 0)

    println(r.shape)
    println(r.data.toList)
  }

  it should "assign to a selection" in {
    val a = NDArray.arange(max = 8).reshape(2, 2, 2)
    val r = NDArray.randn(1)
    a(0, 1, 0) := r
    println(a.data.toList)

    val r2 = NDArray.fill(3.14f, List(2, 2))
    a(1) := r2
    println(a.data.toList)

    // broadcasting
    val r3 = NDArray(100)
    a(0) := r3
    println(a.data.toList)
  }

  it should "narrow" in {
    val a = NDArray.arange(max = 8).reshape(2, 2, 2)
    val b = NDArray.narrow(a, dimension = 0, firstIndex = 1, size = 1)
    NDArray.setValue(b, 3.17f, List(0, 0, 0))
    println(b.shape)
    println(b.data.toList)
    println(a.data.toList)
  }

  it should "linear" in {
    val x = NDArray.randint(1, 5, List(2, 3))
    val y = NDArray.randint(1, 5, List(3, 4))
    val b = NDArray.randint(1, 5, List(2, 4))

    val r = NDArray.linear(x, y, b)
    println(r.shape)
    println(r.data.toList)

    println(r(0, 1))
  }

  //============================
  // numsca tests

  val ta: NDArray = NDArray.arange(max = 10)
  val tb: NDArray = NDArray.reshape(NDArray.arange(max = 9), 3, 3)
  val tc: NDArray = NDArray.reshape(NDArray.arange(max = 2 * 3 * 4), 2, 3, 4)

  // Elements
  it should "retrieve the correct elements" in {
    assert(ta(1).squeeze().data.head == 1)
    assert(tb(1, 0).squeeze().data.head == 3)
    assert(tc(1, 0, 2).squeeze().data.head == 14)

    val i = List(1, 0, 1)
    assert(tc(i: _*).squeeze().data.head == 13)
  }

  it should "change array values in place" in {
    val t = ta.copy()
    t(3) := -5f
    assert(t.data sameElements Array(0, 1, 2, -5, 4, 5, 6, 7, 8, 9))
    t(0) += 7
    println(t.data.toList)
    assert(t.data sameElements Array(7, 1, 2, -5, 4, 5, 6, 7, 8, 9))

    val t2 = tb.copy()
    t2(2, 1) := -7
    t2(1, 2) := -3
    /*
    assert(
      arrayEqual(t2,
        Tensor(0.00, 1.00, 2.00, 3.00, 4.00, -3.00, 6.00, -7.00,
          8.00).reshape(3, 3)))
     */
    assert(t2.shape == List(3, 3))
    assert(
      t2.data.toList == List(0.0, 1.0, 2.0, 3.0, 4.0, -3.0, 6.0, -7.0, 8.0))
  }

  it should "do operations array-wise" in {
    val a2 = ta * 2
    assert(a2.data sameElements Array(0, 2, 4, 6, 8, 10, 12, 14, 16, 18))
  }

  it should "slice over a single dimension" in {

    // turn into a column vector
    val a0 = NDArray.arange(max = 10).reshape(10, 1)

    // A[1:]
    val a1 = a0(1 :>)

    // A[:-1]
    val a2 = a0(0 :> -1)

    // A[1:] - A[:-1]
    val a3 = a1 - a2
    println(a3)
//
//    println(a3.shape)
//    println(a3.data.toList)
//
//    println(a0(5 :>))
//    println(a0(0 :> 5))

  }

}
