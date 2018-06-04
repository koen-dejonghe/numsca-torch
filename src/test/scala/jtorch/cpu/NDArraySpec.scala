package jtorch.cpu

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
    a.size(0) shouldBe 2
    a.size(1) shouldBe 3
    a.data shouldBe Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
  }

  it should "make a zero array" in {
    val shape = List(2, 3)
    val z = NDArray.zeros(shape)
    z.shape shouldBe List(2, 3)
    z.data shouldBe Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  }

  it should "free when no longer used" in {
    val t3 = NDArray.zeros(List(100, 100)) // this one will only get garbage collected at the end of the program
    for (i <- 1 to 100) {
      NDArray.zeros(List(3000, 3000)) // these will get GC'ed as soon as as System.gc() is called
      Thread.sleep(1)
    }
    t3.desc shouldBe "torch.xTensor of size 100x100"
    t3.data.sum shouldBe 0.0
  }

  it should "free in parallel when no longer used" in {
    import scala.concurrent.ExecutionContext.Implicits.global
    import scala.concurrent.duration._

    val t3 = NDArray.zeros(100, 100) // this one will only get garbage collected at the end of the program

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
    t3.data.sum shouldBe 0.0
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
    mean should be (0.0f +- 1e-1f)
    val std = Math.sqrt(data.map(d => Math.pow(d - mean, 2.0)).sum / (data.length - 1))
    println(std)
    std should be (1.0 +- 1e-1)
  }

  it should "randint" in {
    NDArray.setSeed(213L)
    val t = NDArray.randint(high = 10.0, shape=List(10))
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
    val t1 = NDArray.ones(2, 3)
    val t2 = NDArray.ones(1, 3)
    val t3 = NDArray.cmul(t2, t1)
    println(t3.shape)
  }

}
