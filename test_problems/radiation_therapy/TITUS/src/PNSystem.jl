
function AParam( l::Int, k::Int )
    return sqrt( ( ( l - k + 1 ) * ( l + k + 1 ) ) / ( ( 2 * l + 3 ) * ( 2 * l + 1 ) ) );
end

function BParam( l::Int, k::Int ) 
    return sqrt( ( ( l - k ) * ( l + k ) ) / ( ( ( 2 * l + 1 ) * ( 2 * l - 1 ) ) ) );
end

function CParam( l::Int, k::Int ) 
    return sqrt( ( ( l + k + 1 ) * ( l + k + 2 ) ) / ( ( ( 2 * l + 3 ) * ( 2 * l + 1 ) ) ) );
end

function DParam( l::Int, k::Int ) 
    return sqrt( ( ( l - k ) * ( l - k - 1 ) ) / ( ( ( 2 * l + 1 ) * ( 2 * l - 1 ) ) ) );
end

function EParam( l::Int, k::Int ) 
    return sqrt( ( ( l - k + 1 ) * ( l - k + 2 ) ) / ( ( ( 2 * l + 3 ) * ( 2 * l + 1 ) ) ) );
end

function FParam( l::Int, k::Int ) 
    return sqrt( ( ( l + k ) * ( l + k - 1 ) ) / ( ( 2 * l + 1 ) * ( 2 * l - 1 ) ) );
end

function CTilde( l::Int, k::Int ) 
    if k < 0  return 0.0; end
    if k == 0 
        return sqrt( 2 ) * CParam( l, k );
    else
        return CParam( l, k );
    end
end

function DTilde( l::Int, k::Int ) 
    if k < 0  return 0.0; end
    if k == 0 
        return sqrt( 2 ) * DParam( l, k );
    else
        return DParam( l, k );
    end
end

function ETilde( l::Int, k::Int ) 
    if k == 1 
        return sqrt( 2 ) * EParam( l, k );
    else
        return EParam( l, k );
    end
end

function FTilde( l::Int, k::Int ) 
    if k == 1
        return sqrt( 2 ) * FParam( l, k );
    else
        return FParam( l, k );
    end
end

function Sgn( k::Int ) 
    if k >= 0 
        return 1;
    else
        return -1;
    end
end

function GlobalIndex( l::Int, k::Int ) 
    numIndicesPrevLevel  = l * l;    # number of previous indices untill level l-1
    prevIndicesThisLevel = k + l;    # number of previous indices in current level
    return numIndicesPrevLevel + prevIndicesThisLevel;
end

function kPlus( k::Int )  return k + Sgn( k ); end

function kMinus( k::Int )  return k - Sgn( k ); end

function unsigned(x::Float64)
    return Int(floor(x))
end

function int(x::Float64)
    return Int(floor(x))
end

include("PNSystemCPU.jl")
include("PNSystemGPU.jl")